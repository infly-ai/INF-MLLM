from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os
import json
from config.Config import get_config_value
import base64  
from jinja2 import Environment, FileSystemLoader
os.environ["QT_QPA_PLATFORM"] = "xcb"
import random
from PIL import Image

def Jinja_render(template_path, input_data, template, styles, html_path):
    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template(template)
    # Render the template with the data
    rendered_html = template.render(
        language="zh",
        input_data=input_data,
        styles=styles,
    )
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, 'w') as f:
        f.write(rendered_html)

class chrome_render:

    def __init__(self, driver_path="./drive/chromedriver"):
        # Configure Chrome options
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--force-device-scale-factor=1.0")
        self.chrome_options.add_argument("--window-size=794,1123")

        # Set ChromeDriver path
        self.service = Service(executable_path=driver_path)

        # Initialize WebDriver
        self.driver = webdriver.Chrome(service=self.service, options=self.chrome_options)

        # Set window size (A4 paper)
        self.driver.set_window_size(794, 1123)

        
    def close(self):
        self.driver.quit()
        
    def get_location(self, file_path, save_path):
        try:
            self.driver.get(file_path)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

        elements = self.driver.find_elements(By.XPATH, "//*")
        elements_info = []

        for element in elements:
            location = element.location
            size = element.size
            element_info = {
                "Tag": element.tag_name,
                "ID": element.get_attribute('id'),
                "Class": element.get_attribute('class'),
                "Top": location['y'],
                "Left": location['x'],
                "Width": size['width'],
                "Height": size['height']
            }
            elements_info.append(element_info)

        self.driver.execute_script("document.body.style.zoom='100%'")
        cross_column_paragraphs = self.driver.execute_script('''
            const pageData = {
                header: [],
                containerElements: [],
                footer: [],
                pageFootnote: []
            };

            let container = document.querySelector('.main_content');
            if (container) {
                Array.from(container.children).forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) { 
                        if (node.tagName.toLowerCase() === 'h3') { 
                            let range = document.createRange();
                            range.selectNodeContents(node);
                            const titlePosition = range.getBoundingClientRect();

                            pageData.containerElements.push({
                                type: 'section_title',
                                level: 3,
                                content: node.textContent.trim(),
                                position: {
                                    x: titlePosition.left,
                                    y: titlePosition.top,
                                    width: titlePosition.width,
                                    height: titlePosition.height
                                }
                            });

                        } else if (node.tagName.toLowerCase() === 'h2'){
                            let range = document.createRange();
                            range.selectNodeContents(node);
                            const titlePosition = range.getBoundingClientRect();

                            pageData.containerElements.push({
                                type: 'section_title',
                                level: 2,
                                content: node.textContent.trim(),
                                position: {
                                    x: titlePosition.left,
                                    y: titlePosition.top,
                                    width: titlePosition.width,
                                    height: titlePosition.height
                                }
                            });
                        
                        
                        } else if (node.tagName.toLowerCase() === 'h4'){
                        
                            let range = document.createRange();
                            range.selectNodeContents(node);
                            const titlePosition = range.getBoundingClientRect();

                            pageData.containerElements.push({
                                type: 'section_title',
                                level: 4,
                                content: node.textContent.trim(),
                                position: {
                                    x: titlePosition.left,
                                    y: titlePosition.top,
                                    width: titlePosition.width,
                                    height: titlePosition.height
                                }
                            });
                            
                        } else if (node.classList.contains('inline_formula')) {
                        
                            const rect_formula = node.getBoundingClientRect();
                            const latex = node.getAttribute('data-latex');

                            pageData.containerElements.push({
                                type: 'formula',
                                content: latex,
                                position: rect_formula

                            })
                            
                        
                        } else if (node.tagName.toLowerCase() === 'p') {
                        
                                let type = node.classList[0];
                                 
                                if(type==="formula" || type==="inline_formula"){
                                    const rect_formula = node.getBoundingClientRect();
                                    const latex = node.getAttribute('data-latex');

                                    pageData.containerElements.push({
                                        type: 'formula',
                                        content: latex,
                                        position: rect_formula

                                    })
                                }
                                else{
                                
                                
                                    const paragraphRects = Array.from(node.getClientRects()); 

                                    paragraphRects.forEach((rect, i) => {
                                        const isCrossColumn = paragraphRects.length > 1 && rect.width < node.offsetWidth;

                                        let range = document.createRange();
                                        let startRange = document.caretRangeFromPoint(rect.left + 1, rect.top + 1);  
                                        let endRange = document.caretRangeFromPoint(rect.right - 1, rect.bottom - 1); 

                                        if (startRange && endRange) {
                                            if (startRange.startContainer === endRange.endContainer) {
                                                range.setStart(startRange.startContainer, startRange.startOffset);
                                                range.setEnd(endRange.endContainer, endRange.endOffset);
                                            } else {
                                                range.setStart(startRange.startContainer, startRange.startOffset);
                                                range.setEnd(startRange.startContainer, startRange.startContainer.length); 
                                            }

                                            const text = range.toString().trim();
                                            pageData.containerElements.push({
                                                type: type,
                                                isCrossColumn: isCrossColumn, 
                                                part: `part ${i + 1}`,
                                                content: text,  
                                                position: rect
                                            });
                                        }
                                    });
                                
                                }

                        } else if (node.tagName.toLowerCase() === 'img') { 
                            const imgPosition = node.getBoundingClientRect();  

                            pageData.containerElements.push({
                                type: 'figure',
                                src: node.src,
                                alt: node.alt,
                                position: {
                                    x: imgPosition.left,
                                    y: imgPosition.top,
                                    width: imgPosition.width,
                                    height: imgPosition.height
                                }
                            });
                            
                        } else if (node.classList.contains('formula-block')) { 
                        
                        
                                Array.from(node.children).forEach((ele) => {
                                      if (ele.tagName.toLowerCase() === 'p') {
                                            let type = ele.classList[0];

                                            if(type==="formula"){
                                                const rect_formula = ele.getBoundingClientRect();
                                                const latex = ele.getAttribute('data-latex');

                                                pageData.containerElements.push({
                                                    type: 'formula',
                                                    content: latex,
                                                    position: rect_formula
                                                })
                                            }
                                            else{
                                            
                                                const rect_formula_caption = ele.getBoundingClientRect();
                          
                                                pageData.containerElements.push({
                                                    type: 'formula_caption',
                                                    content: ele.textContent.trim(),
                                                    position: rect_formula_caption
                                                })
                                            
                                            
                                            }
                                            
             

                                      } 
                                    });                
 
                        } else if (node.classList.contains('table_outer')) {
                        
                                      Array.from(node.children).forEach((ele) => {
                                      if (ele.tagName.toLowerCase() === 'p') {
                                        let type = ele.classList[0];
                                        const paragraphRects = Array.from(ele.getClientRects());

                                        paragraphRects.forEach((rect, i) => {
                                          const isCrossColumn = paragraphRects.length > 1 && rect.width < ele.offsetWidth;
                                          let range = document.createRange();
                                          let startRange = document.caretRangeFromPoint(rect.left + 1, rect.top + 1); 
                                          let endRange = document.caretRangeFromPoint(rect.right - 1, rect.bottom - 1); 

                                          if (startRange && endRange) {
                                            if (startRange.startContainer === endRange.endContainer) {
                                              range.setStart(startRange.startContainer, startRange.startOffset);
                                              range.setEnd(endRange.endContainer, endRange.endOffset);
                                            } else {
                                              range.setStart(startRange.startContainer, startRange.startOffset);
                                              range.setEnd(startRange.startContainer, startRange.startContainer.length);  
                                            }

                                            const text = range.toString().trim();

                                            pageData.containerElements.push({
                                              type: type,
                                              isCrossColumn: isCrossColumn,
                                              part: `part2 ${i + 1}`,
                                              content: text,  
                                              position: rect
                                            });
                                          }
                                        });

                                      } else {
                                        const tableElement = ele.querySelector('table'); 
                                        if (tableElement) {
                                          const tableRects = tableElement.getClientRects(); 
                                          for (let i = 0; i < tableRects.length; i++) {
                                            const rect = tableRects[i];
                                            let position = {
                                              x: rect.left,
                                              y: rect.top,
                                              width: rect.width,
                                              height: rect.height
                                            };

                                            pageData.containerElements.push({
                                              type: 'table',
                                              content: tableElement.outerHTML,
                                              position: position
                                            });
                                          }
                                        }
                                      }
                                    });

                            
                        } else if( node.classList.contains('unordered') ){
                            const ulElement = node.querySelector('ul');
                            let Position = ulElement.getBoundingClientRect();
                            if (!ulElement) {
                                console.error("can not find <ul>");
                                return; 
                            }
                            const liElements = ulElement.querySelectorAll('li');

                            let mergedContent = "";
                            liElements.forEach(li => {
                                const originalText = li.textContent.trim();
                                const modifiedText = '- ' + JSON.stringify(originalText) + '\\n';
                                mergedContent += modifiedText;

                            });
                            
                            
                            pageData.containerElements.push({
                                    type: 'list-item',  
                                    content: mergedContent.trimEnd(),
                                    position: {
                                        x: Position.left,
                                        y: Position.top,
                                        width: Position.width,
                                        height: Position.height
                                    }
                                });
                            
                        }
                    }
                });
            }
            
            
                function containsContent(text) {
                    return text.trim().length > 0;
                }
            
                function detectHeaderContent() {
                    const headerLeftElement = document.querySelector('.header-left');
                    const headerMidElement = document.querySelector('.header-mid');
                    const headerRightElement = document.querySelector('.header-right');

                    function processHeaderElement(element, positionName) {
                        if (element && containsContent(element.textContent)) {
                            const text = element.textContent.trim();
                            let range = document.createRange();
                            range.selectNodeContents(element);
                            const position = range.getBoundingClientRect();

                            pageData.header.push({
                                type: "header", 
                                content: text,
                                position: {
                                    x: position.left,
                                    y: position.top,
                                    width: position.width,
                                    height: position.height
                                },
                                positionName: positionName,
                            });
                        }
                    }
                    processHeaderElement(headerLeftElement, "left");
                    processHeaderElement(headerMidElement, "mid");
                    processHeaderElement(headerRightElement, "right");
                }
                    function detectFooterContent() {
                        const footerContainers = document.querySelectorAll('.footer-left, .footer-mid, .footer-right');

                        footerContainers.forEach(container => {
                            let range = document.createRange();
                            let elementToMeasure = container; 
                            let content = container.textContent.trim();  
                            const pageNumElement = container.querySelector('.page-num');
                            const circleBackgroundElement = container.querySelector('.circle-background');

                            if (pageNumElement && containsContent(pageNumElement.textContent)) {
                                elementToMeasure = pageNumElement;
                                content = pageNumElement.textContent.trim();
                            } else if (circleBackgroundElement && containsContent(circleBackgroundElement.textContent)) {
                                elementToMeasure = circleBackgroundElement;
                                content = circleBackgroundElement.textContent.trim();
                            } else if (!containsContent(content)) {
                                return;
                            }

                            range.selectNodeContents(elementToMeasure);
                            const position = range.getBoundingClientRect();

                            pageData.footer.push({
                                type: "footer",
                                content: content,
                                position: {
                                    x: position.left,
                                    y: position.top,
                                    width: position.width,
                                    height: position.height
                                },
                                className: elementToMeasure.className
                            });
                        });
                    }


                function detectPageFootnoteContent() {
                    const pageFootnoteElement = document.querySelector('.page_footnote_p');

                    if (pageFootnoteElement && containsContent(pageFootnoteElement.textContent)) {
                        const text = pageFootnoteElement.textContent.trim();
                        let range = document.createRange();
                        range.selectNodeContents(pageFootnoteElement);
                        const position = range.getBoundingClientRect();
                        pageData.pageFootnote.push({
                            type: "page_footnote",
                            content: text,
                            position: {
                                x: position.left,
                                y: position.top,
                                width: position.width,
                                height: position.height
                            }
                        });
                    }
                }

                detectHeaderContent();
                detectFooterContent();
                detectPageFootnoteContent();

            return pageData;

        ''')
        
        overflowDetected = self.driver.execute_script('''
            const elements = document.querySelectorAll('.a4-page *');  
            let isOverflowing = false; 

            elements.forEach(element => {
                const container = element.closest('.a4-page'); 
                const containerRect = container.getBoundingClientRect();  
                const elementRects = element.getClientRects();  

                for (let rect of elementRects) {
                    if (Math.ceil(rect.right) > Math.floor(containerRect.right) || 
                        Math.ceil(rect.bottom) > Math.floor(containerRect.bottom)) {
                        isOverflowing = true; 
                        break;  
                    }
                }
                if (isOverflowing) {
                    return;
                }
            });
            return isOverflowing;  
        ''')
        
        overflowDetected = False


        if not overflowDetected:
            pdf_obj = self.driver.execute_cdp_cmd("Page.printToPDF", {
                "paperWidth": 8.27,       
                "paperHeight": 11.69,
                "marginTop": 0, "marginBottom": 0, "marginLeft": 0, "marginRight": 0,
                "printBackground": True,
                "scale": 1,
                "preferCSSPageSize": True, 
                "landscape": False
            })

            with open(save_path + ".pdf", 'wb') as file:
                file.write(base64.b64decode(pdf_obj['data']))
            
            return cross_column_paragraphs
        else:
            return None