#coding=utf-8
import os
import logging
import torch
import torch.nn as nn
from contextlib import suppress
from einops import rearrange
from transformers import LlamaForCausalLM, LlamaTokenizer
from mmengine import Config
from peft import LoraConfig, get_peft_model
from bigmodelvis import Visualization

from .eva_vit import create_eva_vit_g
from .pooler import Pooler


def get_autocast(precision, cache_enabled=True):
    if precision == "amp":
        return lambda: torch.cuda.amp.autocast(cache_enabled=cache_enabled)
    elif precision == "amp_bfloat16" or precision == "amp_bf16" or precision == 'bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16, cache_enabled=cache_enabled)
    elif precision == 'fp16':
        return lambda: torch.cuda.amp.autocast(dtype=torch.float16, cache_enabled=cache_enabled)
    elif precision == 'fp32':
        return suppress
    else:
        raise ValueError('not supported precision: {}'.format(precision))
    
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

def init_vision_encoder(model_name, 
                        img_size, 
                        drop_path_rate, 
                        use_grad_checkpoint, 
                        vit_adapter_convpass=False,
                        precision="fp32"):
    assert model_name in ["eva_clip_g", "eva2_clip_L", "clip_L", "vit_G"
    ], "vit model must be eva_clip_g, eva2_clip_L or clip_L, or vit_G"
    
    if model_name == "eva_clip_g":
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision,
            vit_adapter_convpass=vit_adapter_convpass,
            cached_file="pretrain_models/eva_vit_g.pth"
        )
    else:
        raise ValueError()
    
    ln_vision = LayerNorm(visual_encoder.num_features)
    return visual_encoder, ln_vision

def is_rank0():
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


class InfMLLM_Inference_LLAMA(nn.Module):
    def __init__(self,
                vit_model: str = "eva_clip_g",
                img_size: int = 224,
                vision_adapter: str = "pooler",
                lm_model: str = "pretrain_models/llama-2-7b-chat-hf",
                lm_tokenizer: str = "pretrain_models/llama-2-7b-chat-hf",
                precision: str = "fp16",
                args=None
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        assert precision in ['bf16', 'amp_bf16']

        ## Initialize visioin enocder
        self.visual_encoder, self.ln_vision = init_vision_encoder(
            vit_model, img_size, drop_path_rate=0.0, use_grad_checkpoint=False, precision="fp16", 
            )

        ## Initialize LM model
        self.lm_tokenizer = LlamaTokenizer.from_pretrained(lm_tokenizer, use_fast=False, trust_remote_code=True)
        self.lm_tokenizer.pad_token = self.lm_tokenizer.unk_token
        self.lm_model = LlamaForCausalLM.from_pretrained(lm_model, trust_remote_code=True,
                                                                torch_dtype=torch.float16 if (precision == 'fp16') else 'auto')
        if hasattr(args, "lm_lora_config") and os.path.isfile(args.lm_lora_config):
            lm_lora_config = Config.fromfile(args.lm_lora_config).lora_config
            lora_config = LoraConfig(
                r=lm_lora_config.lora_r,
                lora_alpha=lm_lora_config.lora_alpha,
                target_modules=lm_lora_config.lora_target_modules,
                lora_dropout=lm_lora_config.lora_dropout,
                bias="none",            # won't use bias currently
                modules_to_save=[],     # TODO: might be helpful if save partial model
                task_type="VL",
            )
            self.lm_model = get_peft_model(self.lm_model, peft_config=lora_config)
            if is_rank0():
                Visualization(self.lm_model).structure_graph()
        
        self.vision_adapter = vision_adapter
        self.vision_prompt_adapter = args.vision_prompt_adapter if hasattr(args, 'vision_prompt_adapter') else False

        if self.vision_adapter == 'pooler':
            self.pooler = Pooler(dim_in=self.visual_encoder.num_features,
                                 dim_out=self.lm_model.config.hidden_size,
                                 pool_out_size=args.pool_out_size)
            self.llama_proj = nn.Identity()
        else:
            raise ValueError()
        
        ## Others
        self.precision = precision
        self._apply_lemmatizer = args.apply_lemmatizer if hasattr(args, 'apply_lemmatizer') else False
        self._lemmatizer = None  
    
    def prompt_wrap(self, img_embeds, atts_img, prompts):
        """
        Replace the placeholder of <ImageHere> with img_embeds.
        Args:
            img_embeds: [batch, 64, 4096]
            atts_img: [batch, 64]
            prompts: ['<Img><ImageHere></Img> ', ...] or ['<image><ImageHere>', ...]
        """
        assert len(img_embeds) == len(atts_img) == len(prompts)

        # add bos
        bos = torch.ones([1, 1], dtype=torch.long, device=img_embeds.device) * self.lm_tokenizer.bos_token_id
        bos_embeds = self.lm_model.get_input_embeddings()(bos)

        emb_lists = []
        image_mask = []
        for each_img_embed, each_prompt in zip(img_embeds, prompts):
            assert '<ImageHere>' in each_prompt
            p_before, p_after = each_prompt.split('<ImageHere>')

            p_before_tokens = self.lm_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.lm_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            
            p_before_embed = self.lm_model.get_input_embeddings()(p_before_tokens.input_ids.long())                                     # [1, 6, 4096]
            p_after_embed = self.lm_model.get_input_embeddings()(p_after_tokens.input_ids.long())                                       # [1, 17, 4096]
            # add 1 bos
            wrapped_emb = torch.cat([bos_embeds, p_before_embed, each_img_embed[None], p_after_embed], dim=1)                           # [1, 87, 4096]
            emb_lists.append(wrapped_emb)

            image_mask.append( torch.tensor([0] * wrapped_emb.size(1)) )
            image_mask[-1][range(bos_embeds.size(1) + p_before_embed.size(1), 
                                     bos_embeds.size(1) + p_before_embed.size(1) + len(each_img_embed))] = 1
            assert image_mask[-1].sum() == each_img_embed.size(0)

        emb_lens = [emb.shape[1] for emb in emb_lists]
        pad_emb = self.lm_model.get_input_embeddings()(torch.tensor(self.lm_tokenizer.pad_token_id, device=img_embeds.device))          # [4096]
        
        assert not self.training 
        # during inference mode, padding on the left
        wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()                                                         # [12, 87, 4096]
        wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)                           # [12, 87]
        wrapped_image_masks = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)                    # [12, 87]
        for i, emb in enumerate(emb_lists):
            wrapped_embs[i, -emb_lens[i]:] = emb
            wrapped_atts[i, -emb_lens[i]:] = 1
            wrapped_image_masks[i, -emb_lens[i]:] = image_mask[i]
        return wrapped_embs, wrapped_atts, wrapped_image_masks

    @torch.no_grad()
    def forward_image_feature(self, image):
        autocast = get_autocast(self.precision, cache_enabled=True)
        with autocast():
            if image.ndim == 4:
                image = image.unsqueeze(1).unsqueeze(1)                                             
            assert image.ndim == 6

            b, t, f = image.shape[:3]
            assert t == 1 and f == 1
            image = rearrange(image, "b t f c h w -> (b t f) c h w")

            image_embeds = self.ln_vision(self.visual_encoder(image))
            
            if self.vision_adapter == 'pooler':
                image_embeds = rearrange(image_embeds, "(b t f) L D -> b t f L D", t=t, f=f)
                query_output= self.pooler(image_embeds)
                query_output = query_output.squeeze(1)
                embeds_img = self.llama_proj(query_output)
            else:
                raise ValueError()
            
            return embeds_img
        
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        autocast = get_autocast(self.precision, cache_enabled=True)
        with autocast():
            image = samples["image"]  
            embeds_img = self.forward_image_feature(image)
            atts_img = torch.ones(embeds_img.size()[:-1], dtype=torch.long).to(image.device)

            prompts = samples["prompts"]
            assert isinstance(prompts, (tuple, list))

            inputs_embeds, attention_mask, masks_img = self.prompt_wrap(embeds_img, atts_img, prompts)
            
            model_args = dict(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.lm_tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            outputs = self.lm_model.generate(**model_args)

            output_text = self.lm_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
                     
            output_text = [text.strip() for text in output_text]

        return output_text
        
    @torch.no_grad()   
    def predict_answers(
        self,
        samples,
        num_beams=5,
        max_len=10,
        min_len=1,
        length_penalty=0,
    ):
        autocast = get_autocast(self.precision, cache_enabled=True)
        with autocast():
            image = samples["image"]  
            embeds_img = self.forward_image_feature(image)
            atts_img = torch.ones(embeds_img.size()[:-1], dtype=torch.long).to(image.device)
            
            prompts = samples["prompts"]
            assert isinstance(prompts, (tuple, list))

            inputs_embeds, attention_mask, masks_img = self.prompt_wrap(embeds_img, atts_img, prompts)

            model_args = dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.lm_tokenizer.eos_token_id,
                length_penalty=length_penalty
            )

            outputs = self.lm_model.generate(**model_args)
            output_text = self.lm_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]

        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]
    
    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
    