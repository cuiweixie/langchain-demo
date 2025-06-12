import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import asyncio

# 加载环境变量
load_dotenv()

class PDFSummarizer:
    def __init__(self, api_key: str, base_url: str = "https://api.proxyxai.com/v1"):
        """
        初始化PDF总结器
        
        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
        """
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model="gpt-3.5-turbo",  # 可以改为gpt-4
            temperature=0.3
        )
        
        self.embeddings = OpenAIEmbeddings(
            api_key=api_key,
            base_url=base_url
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def load_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """
        加载多个PDF文件
        
        Args:
            pdf_paths: PDF文件路径列表
            
        Returns:
            文档列表
        """
        documents = []
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"警告: 文件 {pdf_path} 不存在，跳过...")
                continue
                
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                
                # 为每个文档添加源文件信息
                for doc in docs:
                    doc.metadata['source_file'] = os.path.basename(pdf_path)
                
                documents.extend(docs)
                print(f"成功加载: {pdf_path} ({len(docs)} 页)")
                
            except Exception as e:
                print(f"加载 {pdf_path} 时出错: {str(e)}")
        
        return documents

    def create_summary_report(self, documents: List[Document]) -> str:
        """
        创建总结报告
        
        Args:
            documents: 文档列表
            
        Returns:
            总结报告
        """
        if not documents:
            return "没有找到可处理的文档。"

        # 分割文档
        texts = self.text_splitter.split_documents(documents)
        
        # 创建总结提示模板
        summary_prompt = PromptTemplate(
            template="""
请对以下文档内容进行详细分析和总结，生成一份结构化的报告：

文档内容：
{text}

请按照以下格式生成报告：

# 文档总结报告

## 1. 概述
- 文档数量和来源
- 主要内容类型

## 2. 核心内容摘要
- 关键信息点
- 重要发现
- 主要结论

## 3. 详细分析
- 具体内容分析
- 数据和事实
- 趋势和模式

## 4. 关键洞察
- 重要观点
- 建议和建议
- 潜在影响

## 5. 总结
- 整体结论
- 后续行动建议

请确保报告内容准确、结构清晰、语言简洁明了。
            """,
            input_variables=["text"]
        )

        # 使用map-reduce方式处理长文档
        if len(texts) > 10:  # 如果文档片段太多，使用map-reduce
            return self._create_map_reduce_summary(texts, summary_prompt)
        else:  # 否则使用stuff方式
            return self._create_stuff_summary(texts, summary_prompt)

    def _create_stuff_summary(self, texts: List[Document], prompt: PromptTemplate) -> str:
        """使用stuff方式创建总结"""
        try:
            # 合并所有文本
            combined_text = "\n\n".join([doc.page_content for doc in texts])
            
            # 如果文本太长，截取前面部分
            if len(combined_text) > 15000:
                combined_text = combined_text[:15000] + "\n\n[文档内容过长，已截取前部分进行分析]"
            
            # 调用LLM生成总结
            response = self.llm.invoke(prompt.format(text=combined_text))
            return response.content
            
        except Exception as e:
            return f"生成总结时出错: {str(e)}"

    def _create_map_reduce_summary(self, texts: List[Document], prompt: PromptTemplate) -> str:
        """使用map-reduce方式创建总结"""
        try:
            # 创建map-reduce链
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                map_prompt=PromptTemplate(
                    template="请总结以下内容的要点：\n{text}\n\n要点总结:",
                    input_variables=["text"]
                ),
                combine_prompt=prompt,
                verbose=True
            )
            
            result = chain.invoke({"input_documents": texts})
            return result["output_text"]
            
        except Exception as e:
            return f"生成总结时出错: {str(e)}"

    def create_qa_system(self, documents: List[Document]):
        """
        创建问答系统
        
        Args:
            documents: 文档列表
            
        Returns:
            问答链
        """
        if not documents:
            return None
            
        # 分割文档
        texts = self.text_splitter.split_documents(documents)
        
        # 创建向量存储
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # 创建问答提示
        qa_prompt = PromptTemplate(
            template="""
基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法从提供的文档中找到答案。

上下文：
{context}

问题: {input}

回答:
            """,
            input_variables=["context", "input"]
        )
        
        # 创建问答链
        document_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        qa_chain = create_retrieval_chain(retriever, document_chain)
        
        return qa_chain

def main():
    """主函数"""
    # 设置API密钥
    API_KEY = os.getenv("OPENAI_API_KEY") or "your-api-key-here"
    
    if API_KEY == "your-api-key-here":
        print("请设置OPENAI_API_KEY环境变量或直接修改代码中的API_KEY")
        return
    
    # 初始化PDF总结器
    summarizer = PDFSummarizer(api_key=API_KEY)
    
    # PDF文件路径列表（请修改为实际路径）
    pdf_paths = [
        "demo-data.pdf",
        # 添加更多PDF路径
    ]
    
    print("开始处理PDF文件...")
    
    # 加载PDF文档
    documents = summarizer.load_pdfs(pdf_paths)
    
    if not documents:
        print("没有成功加载任何PDF文档，请检查文件路径。")
        return
    
    print(f"总共加载了 {len(documents)} 个文档页面")
    
    # 生成总结报告
    print("\n正在生成总结报告...")
    summary_report = summarizer.create_summary_report(documents)
    
    # 保存报告
    with open("summary_report.md", "w", encoding="utf-8") as f:
        f.write(summary_report)
    
    print("\n" + "="*50)
    print("总结报告:")
    print("="*50)
    print(summary_report)
    print("\n报告已保存到 summary_report.md")
    

# 异步版本的主函数
async def async_main():
    """异步主函数，用于处理大量文档"""
    API_KEY = os.getenv("OPENAI_API_KEY") or "your-api-key-here"
    
    if API_KEY == "your-api-key-here":
        print("请设置OPENAI_API_KEY环境变量")
        return
    
    summarizer = PDFSummarizer(api_key=API_KEY)
    
    # 这里可以添加异步处理逻辑
    # 例如并行处理多个PDF文件
    pass

if __name__ == "__main__":
    # 创建.env文件示例
    env_example = """
# .env 文件示例
OPENAI_API_KEY=your-actual-api-key-here
    """
    
    print("LangChain PDF总结器")
    print("请确保:")
    print("1. 安装了所需依赖")
    print("2. 设置了正确的API密钥")
    print("3. PDF文件路径正确")
    print("\n.env文件示例:")
    print(env_example)
    
    main()
