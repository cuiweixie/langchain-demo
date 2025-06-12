import os
import pandas as pd
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import re

# 加载环境变量
load_dotenv()

class PersonInfo(BaseModel):
    """定义人员信息的数据结构"""
    name: str = Field(description="人员姓名")
    weight: str = Field(description="体重信息，包含数值和单位")

class PersonInfoList(BaseModel):
    """人员信息列表"""
    persons: List[PersonInfo] = Field(description="提取到的人员信息列表")

class PDFDataExtractor:
    def __init__(self, api_key: str, base_url: str = "https://api.proxyxai.com/v1"):
        """
        初始化PDF数据提取器
        
        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
        """
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model="gpt-3.5-turbo",  # 可以改为 gpt-4
            temperature=0.1
        )
        
        # 设置输出解析器
        self.output_parser = PydanticOutputParser(pydantic_object=PersonInfoList)
        
        # 创建提示词模板
        self.prompt_template = PromptTemplate(
            template="""
你是一个专业的数据提取助手。请从以下文本中提取所有人员的姓名和体重信息。

要求：
1. 仔细识别文本中的人名和对应的体重信息
2. 体重可能以kg、公斤、斤等单位表示
3. 如果没有找到相关信息，返回空列表
4. 确保姓名和体重的对应关系正确

文本内容：
{text}

{format_instructions}

请按照指定格式返回结果：
""",
            input_variables=["text"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )

    def load_pdfs(self, pdf_paths: List[str]) -> List[str]:
        """
        加载多个PDF文件并提取文本
        
        Args:
            pdf_paths: PDF文件路径列表
            
        Returns:
            提取的文本列表
        """
        all_texts = []
        
        for pdf_path in pdf_paths:
            try:
                print(f"正在处理PDF: {pdf_path}")
                
                # 加载PDF
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                
                # 合并所有页面的文本
                full_text = "\n".join([page.page_content for page in pages])
                all_texts.append(full_text)
                
                print(f"成功加载 {pdf_path}，共 {len(pages)} 页")
                
            except Exception as e:
                print(f"加载PDF {pdf_path} 时出错: {str(e)}")
                continue
        
        return all_texts

    def split_text(self, texts: List[str], chunk_size: int = 2000) -> List[str]:
        """
        将长文本分割成较小的块
        
        Args:
            texts: 文本列表
            chunk_size: 每块的大小
            
        Returns:
            分割后的文本块列表
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        
        all_chunks = []
        for text in texts:
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)
        
        return all_chunks

    def extract_person_info(self, text_chunks: List[str]) -> List[Dict]:
        """
        从文本块中提取人员信息
        
        Args:
            text_chunks: 文本块列表
            
        Returns:
            提取的人员信息列表
        """
        all_persons = []
        
        for i, chunk in enumerate(text_chunks):
            try:
                print(f"正在处理文本块 {i+1}/{len(text_chunks)}")
                
                # 构建提示词
                prompt = self.prompt_template.format(text=chunk)
                
                # 调用LLM
                response = self.llm.invoke([HumanMessage(content=prompt)])
                
                # 解析响应
                try:
                    parsed_result = self.output_parser.parse(response.content)
                    
                    # 转换为字典格式
                    for person in parsed_result.persons:
                        person_dict = {
                            "name": person.name,
                            "weight": person.weight
                        }
                        all_persons.append(person_dict)
                        
                except Exception as parse_error:
                    print(f"解析响应时出错: {parse_error}")
                    # 尝试手动解析
                    manual_result = self.manual_parse(response.content)
                    all_persons.extend(manual_result)
                    
            except Exception as e:
                print(f"处理文本块时出错: {str(e)}")
                continue
        
        return all_persons

    def manual_parse(self, response_text: str) -> List[Dict]:
        """
        手动解析响应文本（备用方案）
        
        Args:
            response_text: LLM响应文本
            
        Returns:
            解析出的人员信息列表
        """
        persons = []
        
        # 尝试提取JSON格式的数据
        try:
            # 查找JSON部分
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                if 'persons' in data:
                    for person in data['persons']:
                        if 'name' in person and 'weight' in person:
                            persons.append({
                                'name': person['name'],
                                'weight': person['weight']
                            })
        except:
            pass
        
        return persons

    def remove_duplicates(self, persons: List[Dict]) -> List[Dict]:
        """
        去除重复的人员信息
        
        Args:
            persons: 人员信息列表
            
        Returns:
            去重后的人员信息列表
        """
        seen = set()
        unique_persons = []
        
        for person in persons:
            # 使用姓名作为去重标识
            identifier = person['name'].strip().lower()
            if identifier not in seen and identifier:
                seen.add(identifier)
                unique_persons.append(person)
        
        return unique_persons

    def save_to_csv(self, persons: List[Dict], output_path: str = "extracted_data.csv"):
        """
        将提取的数据保存为CSV文件
        
        Args:
            persons: 人员信息列表
            output_path: 输出文件路径
        """
        if not persons:
            print("没有提取到任何数据")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(persons)
        
        # 保存为CSV
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"数据已保存到: {output_path}")
        print(f"共提取到 {len(persons)} 条记录")
        
        # 显示前几条数据
        print("\n提取的数据预览:")
        print(df.head())

    def process_pdfs(self, pdf_paths: List[str], output_csv: str = "extracted_data.csv"):
        """
        处理PDF文件的主要流程
        
        Args:
            pdf_paths: PDF文件路径列表
            output_csv: 输出CSV文件路径
        """
        print("开始处理PDF文件...")
        
        # 1. 加载PDF文件
        texts = self.load_pdfs(pdf_paths)
        if not texts:
            print("没有成功加载任何PDF文件")
            return
        
        # 2. 分割文本
        print("正在分割文本...")
        text_chunks = self.split_text(texts)
        print(f"文本已分割为 {len(text_chunks)} 个块")
        
        # 3. 提取人员信息
        print("正在提取人员信息...")
        persons = self.extract_person_info(text_chunks)
        
        # 4. 去重
        unique_persons = self.remove_duplicates(persons)
        print(f"去重后剩余 {len(unique_persons)} 条记录")
        
        # 5. 保存为CSV
        self.save_to_csv(unique_persons, output_csv)

def main():
    """主函数"""
    # 设置API密钥
    API_KEY = os.getenv("OPENAI_API_KEY")  # 从环境变量读取
    if not API_KEY:
        API_KEY = "your-api-key-here"  # 或直接设置
    
    # 创建提取器
    extractor = PDFDataExtractor(api_key=API_KEY)
    
    # 设置PDF文件路径（可以是单个或多个文件）
    pdf_files = [
        "demo-data.pdf",
        # 添加更多PDF文件路径
    ]
    
    # 检查文件是否存在
    existing_files = []
    for pdf_file in pdf_files:
        if Path(pdf_file).exists():
            existing_files.append(pdf_file)
        else:
            print(f"警告: 文件不存在 - {pdf_file}")
    
    if not existing_files:
        print("没有找到有效的PDF文件")
        return
    
    # 处理PDF文件
    extractor.process_pdfs(existing_files, "person_weight_data.csv")

if __name__ == "__main__":
    main()
