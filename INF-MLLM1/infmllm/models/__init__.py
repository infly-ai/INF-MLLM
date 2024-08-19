from .infmllm_inference_llama import InfMLLM_Inference_LLAMA


def build_model(model_type,
                vit_model: str = "eva_clip_g",
                img_size: int = 224,
                vision_adapter: str = "pooler",
                lm_model: str = "pretrain_models/llama-2-7b-chat-hf/",
                lm_tokenizer: str = "pretrain_models/llama-2-7b-chat-hf/",
                precision: str = "bf16",
                args=None):
    
    if model_type.lower() == 'infmllm_inference_llama':
        model = InfMLLM_Inference_LLAMA(
            vit_model=vit_model,
            img_size=img_size,
            vision_adapter=vision_adapter,
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            precision=precision,
            args=args
        )
    else:
        raise ValueError()
    
    return model