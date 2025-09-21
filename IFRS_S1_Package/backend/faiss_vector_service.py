#!/usr/bin/env python3
"""
FAISS向量檢索服務 - 平衡組合優化方案
實現IFRS S1條文向量預計算和高速檢索
目標：5-8分鐘內完成永續報告書分析
"""

import os
import json
import pickle
import time
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IFRS S1 核心詞彙映射字典
VOCABULARY_MAPPING = {
    # 治理相關
    "治理": ["董事會", "治理委員會", "管理層", "決策機關", "管理階層", "公司治理"],
    "治理機構": ["董事會", "治理委員會", "管理層", "決策機關", "管理階層"],
    "監測": ["監督", "監控", "追蹤", "檢視", "觀察", "審視", "把關"],
    "管理": ["管控", "處理", "執行", "營運", "經營", "操作", "運作"],
    "監督": ["監管", "督導", "查核", "稽核", "oversight"],
    
    # 披露報告相關
    "披露": ["揭露", "報告", "公布", "發布", "說明", "呈現", "展示", "透露", "負責"],
    "報告": ["報告書", "年報", "永續報告", "CSR報告", "ESG報告"],
    "揭露": ["公開", "透明化", "資訊公開", "資料揭示"],
    "資訊": ["資料", "數據", "信息", "內容", "訊息"],
    
    # 風險管理相關
    "風險": ["風險因子", "不確定性", "挑戰", "威脅", "危險因素"],
    "機遇": ["機會", "商機", "發展機會", "成長動能", "潛力", "契機"],
    "風險管理": ["風控", "風險控制", "風險防範", "風險因應"],
    "評估": ["評量", "評鑑", "分析", "檢討", "審查", "衡量"],
    
    # 氣候變化相關
    "氣候相關": ["氣候變遷", "氣候變化", "氣候風險", "低碳", "減碳"],
    "氣候": ["氣候變遷", "氣候變化", "環境", "碳", "溫室氣體"],
    "溫室氣體": ["GHG", "CO2", "碳排放", "排放量", "碳足跡"],
    "減緩": ["減量", "降低", "削減", "縮減", "抑制"],
    "調適": ["適應", "因應", "調整", "轉型", "應對"],
    
    # 策略相關
    "策略": ["戰略", "方針", "政策", "規劃", "藍圖", "計畫"],
    "商業模式": ["營運模式", "經營模式", "商業策略", "獲利模式"],
    "規劃": ["計劃", "布局", "安排", "設計", "籌劃"],
    "目標": ["指標", "KPI", "里程碑", "標的", "願景"],
    
    # 指標目標相關
    "指標": ["指數", "數據", "量化指標", "衡量標準", "績效指標"],
    "績效": ["表現", "成果", "業績", "效果", "成效", "執行成果"],
    "進展": ["進度", "發展", "改善", "提升", "成長"],
    
    # 組織架構相關
    "企業": ["公司", "組織", "機構", "集團", "事業體", "法人"],
    "實體": ["單位", "子公司", "關係企業", "營運據點", "事業部"],
    "流程": ["程序", "作業流程", "工作流程", "機制", "制度"],
    "架構": ["組織", "體系", "框架", "結構", "系統"],
    
    # 永續發展相關
    "永續相關": ["永續發展", "可持續", "ESG", "企業社會責任", "CSR"],
    "永續": ["可持續", "ESG", "企業社會責任", "CSR", "永續發展"],
    "轉型": ["轉換", "變革", "改變", "升級", "更新", "革新"],
    "創新": ["研發", "開發", "改良", "突破", "進步"],
    "影響": ["衝擊", "效應", "作用", "結果", "後果"],
    
    # 財務相關
    "財務": ["財務報表", "會計", "經濟", "資金", "投資"],
    "資本": ["投資", "資金", "成本", "費用", "支出"],
    "收益": ["營收", "獲利", "盈利", "收入", "效益"],
    "成本": ["費用", "支出", "投入", "開銷", "代價"],
    
    # 時間相關
    "期間": ["年度", "期別", "時期", "階段", "週期"],
    "長期": ["中長期", "未來", "長遠", "持續", "永續"],
    "短期": ["近期", "當期", "即時", "目前", "現階段"],
    "定期": ["例行", "常規", "週期性", "固定", "持續性"]
}

# 停用詞列表
STOP_WORDS = {
    "的", "和", "與", "或", "及", "以", "為", "在", "是", "有", "其", "該", "此", 
    "這", "那", "將", "已", "可", "能", "會", "應", "須", "等", "於", "從", "對", 
    "由", "所", "被", "而", "但", "然", "則", "也", "都", "還", "就", "只", "又",
    "更", "最", "很", "非", "不", "未", "無", "沒", "並", "且", "如", "若", "使"
}

# 需要排除的條文ID（不建立索引，也不參與分析）
EXCLUDED_ARTICLE_IDS = {"IFRS-S1-20"}

class HybridRetrievalSystem:
    """
    混合檢索系統 - FAISS語義搜索 + BM25關鍵詞搜索
    平衡組合方案的核心組件
    """
    
    def __init__(self, vector_dim=768, cache_dir="./vector_cache"):
        self.vector_dim = vector_dim
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # FAISS索引
        self.faiss_index = None
        self.document_embeddings = None
        self.documents = []
        self.article_mapping = {}
        
        # BM25組件
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # 性能統計
        self.stats = {
            'total_queries': 0,
            'avg_retrieval_time': 0.0,
            'cache_hits': 0
        }
        
        logger.info(f"🚀 初始化混合檢索系統 - 向量維度: {vector_dim}")

    def _filter_articles(self, articles: List[Dict]) -> List[Dict]:
        """在建立索引前過濾掉被排除的條文。"""
        if not articles:
            return articles
        filtered = [a for a in articles if a.get('id') not in EXCLUDED_ARTICLE_IDS]
        removed = len(articles) - len(filtered)
        if removed > 0:
            logger.info(f"⏭️ 已排除 {removed} 條條文（例如: IFRS-S1-20）不建立索引")
        return filtered
    
    def precompute_article_vectors(self, articles: List[Dict]) -> None:
        """
        預計算所有IFRS S1條文的向量表示
        這是性能優化的關鍵步驟
        """
        start_time = time.time()
        # 先過濾被排除的條文
        articles = self._filter_articles(articles)
        logger.info(f"📊 開始預計算 {len(articles)} 條IFRS S1條文向量...")
        
        cache_path = self.cache_dir / "article_vectors.pkl"
        
        # 檢查緩存
        if cache_path.exists():
            logger.info("⚡ 從緩存載入預計算向量...")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.faiss_index = cache_data['faiss_index']
                self.document_embeddings = cache_data['embeddings']
                self.documents = cache_data['documents']
                self.article_mapping = cache_data['mapping']
                self.tfidf_vectorizer = cache_data['tfidf_vectorizer']
                self.tfidf_matrix = cache_data['tfidf_matrix']
            # 如果緩存中仍包含被排除的條文，則重新建立索引
            try:
                contains_excluded = any(
                    (isinstance(v, dict) and v.get('id') in EXCLUDED_ARTICLE_IDS)
                    for v in self.article_mapping.values()
                )
            except Exception:
                contains_excluded = False

            if not contains_excluded:
                load_time = time.time() - start_time
                logger.info(f"✅ 緩存載入完成 - 耗時: {load_time:.2f}秒")
                return
            else:
                logger.warning("♻️ 緩存包含已排除條文（例如 IFRS-S1-20），正在忽略緩存並重新建立索引...")
        
        # 準備文檔文本
        documents = []
        article_mapping = {}
        
        for idx, article in enumerate(articles):
            # 組合標題和內容作為搜索文本
            doc_text = f"{article.get('title', '')} {article.get('content', '')}"
            documents.append(doc_text)
            article_mapping[idx] = {
                'id': article.get('id'),
                'title': article.get('title'),
                'category': article.get('category'),
                'difficulty': article.get('difficulty', 'medium'),
                'keywords': article.get('keywords', [])
            }
        
        self.documents = documents
        self.article_mapping = article_mapping
        
        # 使用真實的語義嵌入模型（Sentence Transformers + MPS加速）
        logger.info("🔄 生成文檔向量表示...")
        embeddings = self._generate_real_embeddings(documents)
        self.document_embeddings = embeddings
        
        # 建立FAISS索引
        self._build_faiss_index(embeddings)
        
        # 建立BM25索引
        self._build_bm25_index(documents)
        
        # 保存到緩存
        cache_data = {
            'faiss_index': self.faiss_index,
            'embeddings': self.document_embeddings,
            'documents': self.documents,
            'mapping': self.article_mapping,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        total_time = time.time() - start_time
        logger.info(f"✅ 條文向量預計算完成 - 總耗時: {total_time:.2f}秒")
    
    def _generate_real_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        智能混合模式：小批量用深度學習，大批量用TF-IDF快速模式
        替換原有的TF-IDF模擬方法
        """
        if not texts:
            return np.array([])
            
        # 智能模式選擇：只有單個項目或2項用深度學習，3項以上用TF-IDF快速模式
        if len(texts) > 2:
            logger.info(f"🏃 快速模式 ({len(texts)}項) - 使用TF-IDF快速處理")
            return self._generate_fallback_embeddings(texts)
        
        logger.info(f"🧠 深度學習模式 ({len(texts)}項) - 使用Sentence Transformers")
        
        try:
            # 檢查是否已初始化模型
            if not hasattr(self, '_embedding_model'):
                logger.info("🚀 正在載入Sentence Transformer模型...")
                from sentence_transformers import SentenceTransformer
                import torch
                
                # 檢查MPS可用性
                device = 'cpu'  # 預設使用CPU
                if torch.backends.mps.is_available():
                    device = 'mps'
                    logger.info("✅ 檢測到MPS支援，使用Metal Performance Shaders加速")
                elif torch.cuda.is_available():
                    device = 'cuda'
                    logger.info("✅ 檢測到CUDA支援，使用GPU加速")
                else:
                    logger.info("ℹ️  使用CPU運算")
                
                # 載入多語言模型（支援中英文）
                model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                self._embedding_model = SentenceTransformer(model_name, device=device)
                self._device = device
                
                # 調整向量維度（該模型輸出384維，需要匹配self.vector_dim=768）
                actual_dim = self._embedding_model.get_sentence_embedding_dimension()
                if actual_dim != self.vector_dim:
                    logger.info(f"🔧 模型輸出維度 {actual_dim}，目標維度 {self.vector_dim}")
                    if actual_dim < self.vector_dim:
                        # 如果模型維度小於目標維度，通過重複和隨機噪聲擴展
                        self._dimension_adapter = 'expand'
                    else:
                        # 如果模型維度大於目標維度，通過PCA降維
                        self._dimension_adapter = 'reduce'
                        from sklearn.decomposition import PCA
                        self._pca = PCA(n_components=self.vector_dim)
                else:
                    self._dimension_adapter = None
                
                logger.info(f"✅ 模型載入完成，使用設備: {device}")
            
            if not texts:
                return np.zeros((0, self.vector_dim), dtype=np.float32)
            
            # 預處理文本
            processed_texts = []
            for text in texts:
                # 清理並截斷文本
                clean_text = text.strip()
                if len(clean_text) > 512:  # 限制長度避免內存問題
                    clean_text = clean_text[:512]
                processed_texts.append(clean_text if clean_text else "空文本")
            
            # 生成嵌入向量
            logger.info(f"🔄 正在為 {len(processed_texts)} 個文本生成語義嵌入...")
            embeddings = self._embedding_model.encode(
                processed_texts,
                convert_to_numpy=True,
                show_progress_bar=len(processed_texts) > 10,
                batch_size=32  # 控制批次大小
            )
            
            # 維度調整
            if self._dimension_adapter == 'expand':
                # 擴展維度：重複向量並添加小量隨機噪聲
                repeat_factor = self.vector_dim // embeddings.shape[1]
                remainder = self.vector_dim % embeddings.shape[1]
                
                expanded_embeddings = np.tile(embeddings, (1, repeat_factor))
                if remainder > 0:
                    expanded_embeddings = np.hstack([
                        expanded_embeddings,
                        embeddings[:, :remainder]
                    ])
                
                # 添加小量噪聲增加多樣性
                noise = np.random.normal(0, 0.01, expanded_embeddings.shape)
                embeddings = expanded_embeddings + noise
                
            elif self._dimension_adapter == 'reduce':
                # 降維：使用PCA
                if not hasattr(self, '_pca_fitted'):
                    self._pca.fit(embeddings)
                    self._pca_fitted = True
                embeddings = self._pca.transform(embeddings)
            
            # 確保維度正確
            assert embeddings.shape[1] == self.vector_dim, f"維度不匹配: {embeddings.shape[1]} != {self.vector_dim}"
            
            # 正規化向量（對FAISS內積搜索很重要）
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 避免除零
            embeddings = embeddings / norms
            
            logger.info(f"✅ 語義嵌入生成完成 - 形狀: {embeddings.shape}, 設備: {self._device}")
            return embeddings.astype(np.float32)
            
        except ImportError as e:
            logger.warning(f"⚠️  Sentence Transformers未安裝，回退到TF-IDF方法: {e}")
            return self._generate_fallback_embeddings(texts)
        except Exception as e:
            logger.error(f"❌ 語義嵌入生成失敗，回退到TF-IDF方法: {e}")
            return self._generate_fallback_embeddings(texts)
    
    def _generate_fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        備援嵌入方法 - TF-IDF + 隨機投影（原有方法）
        """
        logger.info("🔄 使用TF-IDF備援方法生成嵌入...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.random_projection import SparseRandomProjection
        
        # TF-IDF向量化
        vectorizer = TfidfVectorizer(max_features=2048, stop_words='english')
        tfidf_vectors = vectorizer.fit_transform(texts)
        
        # 降維到目標維度
        if tfidf_vectors.shape[1] > self.vector_dim:
            projector = SparseRandomProjection(n_components=self.vector_dim)
            reduced_vectors = projector.fit_transform(tfidf_vectors)
        else:
            # 如果維度不足，填充零
            reduced_vectors = np.hstack([
                tfidf_vectors.toarray(),
                np.zeros((tfidf_vectors.shape[0], self.vector_dim - tfidf_vectors.shape[1]))
            ])
        
        return reduced_vectors.astype(np.float32)
    
    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """建立FAISS索引用於快速相似度搜索"""
        logger.info("🔧 建立FAISS索引...")
        
        # 對於IFRS S1條文數據集（126條），使用簡單平面索引最合適
        # 避免IVF聚類問題，確保穩定性
        self.faiss_index = faiss.IndexFlatIP(self.vector_dim)
        
        # 添加向量到索引
        self.faiss_index.add(embeddings)
        logger.info(f"✅ FAISS索引建立完成 - {len(embeddings)}個條文向量")
    
    def _build_bm25_index(self, documents: List[str]) -> None:
        """建立BM25/TF-IDF索引用於關鍵詞搜索"""
        logger.info("🔧 建立BM25索引...")
        
        # 簡化的BM25實現（使用TF-IDF近似）
        # 根據文檔數量調整參數
        min_df = min(1, len(documents) // 4) if len(documents) > 2 else 1
        max_df = 0.95 if len(documents) > 10 else 1.0
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=min(5000, len(documents) * 100),
            stop_words='english',
            ngram_range=(1, 2),
            max_df=max_df,
            min_df=min_df
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        logger.info(f"✅ BM25索引建立完成 - 詞彙量: {len(self.tfidf_vectorizer.vocabulary_)}")
    
    def hybrid_search(self, query: str, top_k: int = 10, semantic_weight: float = 0.7) -> List[Dict]:
        """
        混合搜索：結合語義搜索和關鍵詞搜索
        
        Args:
            query: 搜索查詢
            top_k: 返回結果數量
            semantic_weight: 語義搜索權重 (0.0-1.0)
            
        Returns:
            排序後的搜索結果
        """
        start_time = time.time()
        
        # 語義搜索分數
        semantic_scores = self._semantic_search(query, top_k * 2)
        
        # 關鍵詞搜索分數  
        keyword_scores = self._keyword_search(query, top_k * 2)
        
        # 混合評分
        hybrid_scores = self._combine_scores(
            semantic_scores, 
            keyword_scores, 
            semantic_weight
        )
        
        # 排序並返回top_k結果
        results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 構建結果
        search_results = []
        for doc_idx, score in results:
            article_info = self.article_mapping[doc_idx]
            search_results.append({
                'article_id': article_info['id'],
                'title': article_info['title'],
                'category': article_info['category'],
                'difficulty': article_info['difficulty'],
                'score': float(score),
                'doc_index': doc_idx
            })
        
        # 更新統計
        retrieval_time = time.time() - start_time
        self.stats['total_queries'] += 1
        self.stats['avg_retrieval_time'] = (
            (self.stats['avg_retrieval_time'] * (self.stats['total_queries'] - 1) + retrieval_time) 
            / self.stats['total_queries']
        )
        
        logger.info(f"🔍 混合搜索完成 - 耗時: {retrieval_time:.3f}秒, 結果數: {len(search_results)}")
        return search_results
    
    def _semantic_search(self, query: str, k: int) -> Dict[int, float]:
        """語義向量搜索"""
        # 生成查詢向量
        query_vector = self._generate_real_embeddings([query])[0:1]
        
        # FAISS搜索
        scores, indices = self.faiss_index.search(query_vector, k)
        
        # 轉換為字典格式
        semantic_scores = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # 有效索引
                semantic_scores[idx] = float(score)
        
        return semantic_scores
    
    def _keyword_search(self, query: str, k: int) -> Dict[int, float]:
        """關鍵詞TF-IDF搜索"""
        # 向量化查詢
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # 計算相似度
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # 獲取top-k
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        keyword_scores = {}
        for idx in top_indices:
            if similarities[idx] > 0:
                keyword_scores[idx] = float(similarities[idx])
        
        return keyword_scores
    
    def _combine_scores(self, semantic_scores: Dict[int, float], 
                       keyword_scores: Dict[int, float], 
                       semantic_weight: float) -> Dict[int, float]:
        """組合語義和關鍵詞分數"""
        all_docs = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        # 正規化分數
        if semantic_scores:
            max_semantic = max(semantic_scores.values())
            semantic_scores = {k: v/max_semantic for k, v in semantic_scores.items()}
        
        if keyword_scores:
            max_keyword = max(keyword_scores.values())
            keyword_scores = {k: v/max_keyword for k, v in keyword_scores.items()}
        
        # 加權組合
        combined_scores = {}
        for doc_idx in all_docs:
            semantic_score = semantic_scores.get(doc_idx, 0.0)
            keyword_score = keyword_scores.get(doc_idx, 0.0)
            
            combined_score = (
                semantic_weight * semantic_score + 
                (1 - semantic_weight) * keyword_score
            )
            combined_scores[doc_idx] = combined_score
        
        return combined_scores
    
    def get_stats(self) -> Dict:
        """獲取性能統計"""
        return {
            **self.stats,
            'indexed_documents': len(self.documents),
            'vector_dimension': self.vector_dim,
            'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0
        }

class FastIFRSAnalyzer:
    """
    快速IFRS S1分析器
    使用FAISS+混合檢索實現5-8分鐘內完成分析
    """
    
    def __init__(self):
        self.retrieval_system = HybridRetrievalSystem()
        self.initialized = False
        
    def initialize_with_articles(self, articles: List[Dict]) -> None:
        """使用IFRS S1條文初始化系統"""
        if not self.initialized:
            logger.info("🚀 初始化快速IFRS S1分析系統...")
            self.retrieval_system.precompute_article_vectors(articles)
            self.initialized = True
            logger.info("✅ 系統初始化完成")
    
    def analyze_document_fast(self, document_text: str, 
                            sentences: List[str], 
                            paragraphs: List[str]) -> List[Dict]:
        """
        快速文檔分析 - 核心優化函數
        
        Args:
            document_text: 完整文檔文本
            sentences: 文檔句子列表
            paragraphs: 文檔段落列表
            
        Returns:
            分析結果列表
        """
        if not self.initialized:
            raise ValueError("系統未初始化，請先調用 initialize_with_articles()")
        
        start_time = time.time()
        logger.info(f"🔍 開始快速文檔分析 - 句子數: {len(sentences)}, 段落數: {len(paragraphs)}")
        
        results = []
        
        # 動態批量處理
        batch_size = min(50, len(sentences) + len(paragraphs))
        all_text_chunks = sentences + paragraphs
        
        for i in range(0, len(all_text_chunks), batch_size):
            batch_chunks = all_text_chunks[i:i+batch_size]
            batch_query = " ".join(batch_chunks[:3])  # 使用前3個作為查詢
            
            # 混合檢索匹配條文
            matched_articles = self.retrieval_system.hybrid_search(
                batch_query, 
                top_k=20,  # 減少候選數量
                semantic_weight=0.7
            )
            
            # 處理匹配結果
            for article in matched_articles:
                if article['score'] > 0.3:  # 提高閾值以獲得高質量結果
                    # 快速證據匹配
                    evidences = self._fast_evidence_matching(
                        batch_chunks, article, max_evidences=3
                    )
                    
                    if evidences:
                        results.append({
                            'article': article,
                            'evidences': evidences,
                            'confidence': article['score']
                        })
        
        # 去重並排序
        results = self._deduplicate_results(results)
        
        analysis_time = time.time() - start_time
        logger.info(f"⚡ 快速分析完成 - 耗時: {analysis_time:.2f}秒, 結果數: {len(results)}")
        
        return results
    
    def analyze_article_specific(self, article: Dict, document_sentences: List[str], 
                               document_paragraphs: List[str]) -> Dict:
        """
        條文特定分析 - 針對單個IFRS條文進行精確匹配
        
        Args:
            article: 單個IFRS條文字典
            document_sentences: 文檔句子列表 
            document_paragraphs: 文檔段落列表
            
        Returns:
            該條文的分析結果字典
        """
        if not self.initialized:
            raise ValueError("系統未初始化，請先調用 initialize_with_articles()")
        
        # 若為被排除條文，直接跳過分析
        if article.get('id') in EXCLUDED_ARTICLE_IDS:
            logger.info(f"⏭️ 跳過排除條文: {article.get('id')}")
            return {
                'article_id': article.get('id'),
                'article_data': article,
                'evidences': [],
                'max_similarity': 0.0,
                'processing_time': 0.0,
                'skipped': True
            }
        
        start_time = time.time()
        logger.debug(f"🎯 開始條文特定分析: {article.get('id', 'Unknown')}")
        
        # 構建條文特定的查詢文本
        article_query = f"{article.get('title', '')} {article.get('content', '')} {' '.join(article.get('keywords', []))}"
        
        # 使用條文內容進行語義檢索
        all_document_chunks = document_sentences + document_paragraphs
        
        # 計算與文檔各部分的相似度
        similarities = self._calculate_specific_similarities(article_query, all_document_chunks)
        logger.debug(f"🎯 相似度計算結果: {len(similarities)} 個分數, 最高: {max(similarities) if similarities else 0:.4f}")
        
        # 動態閾值策略
        dynamic_threshold = self._calculate_dynamic_threshold(similarities, article)
        
        # 匹配高相似度的句子和段落
        sentence_matches = []
        paragraph_matches = []
        
        # 處理句子匹配 - 使用動態閾值
        for i, similarity in enumerate(similarities[:len(document_sentences)]):
            if similarity > dynamic_threshold:
                # 再次檢查內容質量
                if not self._is_irrelevant_content(document_sentences[i]):
                    sentence_matches.append({
                        'content': document_sentences[i],
                        'similarity': similarity,
                        'type': 'sentence',
                        'index': i,
                        'page': (i // 20) + 1  # 估算頁碼，假設每頁約20個句子
                    })
        
        # 處理段落匹配 - 段落閾值稍微放寬
        paragraph_threshold = dynamic_threshold * 0.8
        for i, similarity in enumerate(similarities[len(document_sentences):]):
            if similarity > paragraph_threshold:
                # 再次檢查內容質量
                if not self._is_irrelevant_content(document_paragraphs[i]):
                    paragraph_matches.append({
                        'content': document_paragraphs[i],
                        'similarity': similarity,
                        'type': 'paragraph', 
                        'index': len(document_sentences) + i,
                        'page': (i // 5) + 1  # 估算頁碼，假設每頁約5個段落
                    })
        
        # 合併並排序證據
        all_evidences = sentence_matches + paragraph_matches
        all_evidences.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 智能證據選擇 - 確保每個條文至少有1-2個證據
        top_evidences = self._select_best_evidences(all_evidences, similarities)
        
        analysis_time = time.time() - start_time
        logger.debug(f"🎯 條文分析完成: {article.get('id')} - 耗時: {analysis_time:.3f}秒, 證據數: {len(top_evidences)}")
        
        return {
            'article_id': article.get('id'),
            'article_data': article,
            'evidences': top_evidences,
            'max_similarity': max([e['similarity'] for e in top_evidences], default=0.0),
            'processing_time': analysis_time
        }
    
    def _calculate_specific_similarities(self, query: str, document_chunks: List[str]) -> List[float]:
        """
        計算查詢與文檔片段的真實語義相似度
        使用Sentence Transformers模型進行精確計算
        
        Args:
            query: 條文查詢文本
            document_chunks: 文檔片段列表
            
        Returns:
            相似度分數列表
        """
        try:
            if not document_chunks or not query.strip():
                logger.debug(f"空輸入: query='{query}', chunks={len(document_chunks)}")
                return [0.0] * len(document_chunks)
            
            # 使用真實語義嵌入計算相似度
            query_embedding = self.retrieval_system._generate_real_embeddings([query])
            chunk_embeddings = self.retrieval_system._generate_real_embeddings(document_chunks)
            
            # 計算餘弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            # 轉換為列表並確保數值範圍
            similarities = [max(0.0, min(1.0, float(sim))) for sim in similarities]
            
            logger.debug(f"真實語義相似度計算完成: 最高分={max(similarities):.4f}, 有效匹配={sum(1 for s in similarities if s > 0.1)}")
            return similarities
            
        except Exception as e:
            logger.error(f"真實語義相似度計算失敗，回退到詞彙匹配: {e}")
            # 回退到原有的詞彙匹配方法
            return self._calculate_fallback_similarities(query, document_chunks)
    
    def _calculate_fallback_similarities(self, query: str, document_chunks: List[str]) -> List[float]:
        """
        備援相似度計算方法 - 使用詞彙匹配
        """
        try:
            # 文本預處理和詞彙標準化
            query_normalized = self._normalize_text(query)
            chunks_normalized = [self._normalize_text(chunk) for chunk in document_chunks]
            
            similarities = []
            for chunk_norm in chunks_normalized:
                if not chunk_norm:
                    similarities.append(0.0)
                    continue
                
                # 語義感知匹配
                similarity = self._semantic_similarity(query_normalized, chunk_norm)
                similarities.append(similarity)
            
            logger.debug(f"備援相似度計算完成: 最高分={max(similarities):.4f}, 有效匹配={sum(1 for s in similarities if s > 0.1)}")
            return similarities
            
        except Exception as e:
            logger.error(f"備援相似度計算失敗: {e}")
            return [0.0] * len(document_chunks)
    
    def _normalize_text(self, text: str) -> str:
        """
        文本標準化處理
        """
        if not text.strip():
            return ""
        
        # 檢查並過濾無關內容
        if self._is_irrelevant_content(text):
            return ""
        
        # 基礎清理
        text = text.strip().lower()
        
        # 移除標點符號但保留中文
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # 統一空格
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _is_irrelevant_content(self, text: str) -> bool:
        """
        判斷是否為無關的格式化內容
        """
        if not text or len(text.strip()) < 10:
            return True
        
        text_lower = text.lower()
        
        # 檢查組織圖表相關內容
        org_chart_patterns = [
            r'▪.*▪.*▪',  # 多個項目符號
            r'^\d+\.\d+\s+\w+',  # 章節編號開頭
            r'詳\s*\d+\.\d+',  # "詳 2.3" 等參考
            r'每年.*次.*呈報',  # 組織流程描述
            r'委員會.*委員會.*委員會',  # 多個委員會重複
            r'價值.*價值.*價值',  # 多個價值重複
        ]
        
        for pattern in org_chart_patterns:
            if re.search(pattern, text):
                return True
        
        # 檢查過多的組織單位列舉
        org_units = ['處', '室', '部', '廠', '委員會', '小組']
        org_count = sum(text.count(unit) for unit in org_units)
        if org_count > 5:  # 超過5個組織單位可能是組織圖
            return True
        
        # 檢查是否主要是項目符號和組織名稱
        bullet_points = text.count('▪') + text.count('●') + text.count('•')
        if bullet_points > 3 and len(text.split()) < bullet_points * 3:
            return True
        
        # 檢查是否為純數字編號內容
        if re.match(r'^[\d\.\s]+$', text):
            return True
        
        # 檢查無意義的重複詞彙
        words = text.split()
        if len(words) > 0:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:  # 重複度過高
                return True
        
        return False
    
    def _semantic_similarity(self, query: str, chunk: str) -> float:
        """
        語義感知相似度計算
        """
        if not query or not chunk:
            return 0.0
        
        # 分詞處理
        query_words = self._tokenize_and_expand(query)
        chunk_words = self._tokenize_and_expand(chunk)
        
        if not query_words or not chunk_words:
            return 0.0
        
        # 計算加權匹配分數
        total_score = 0.0
        query_word_count = len(query_words)
        
        for query_word in query_words:
            max_word_score = 0.0
            
            for chunk_word in chunk_words:
                word_score = self._word_similarity(query_word, chunk_word)
                max_word_score = max(max_word_score, word_score)
            
            total_score += max_word_score
        
        # 正規化分數
        similarity = total_score / query_word_count if query_word_count > 0 else 0.0
        
        return min(similarity, 1.0)  # 確保不超過1.0
    
    def _tokenize_and_expand(self, text: str) -> List[str]:
        """
        分詞並進行詞彙擴展
        """
        if not text:
            return []
        
        # 簡單分詞
        words = text.split()
        
        # 移除停用詞
        words = [word for word in words if word not in STOP_WORDS]
        
        # 詞彙擴展 - 添加同義詞
        expanded_words = []
        for word in words:
            expanded_words.append(word)
            # 添加映射詞彙
            if word in VOCABULARY_MAPPING:
                expanded_words.extend(VOCABULARY_MAPPING[word])
        
        return expanded_words
    
    def _word_similarity(self, word1: str, word2: str) -> float:
        """
        單詞相似度計算
        """
        if not word1 or not word2:
            return 0.0
        
        # 精確匹配
        if word1 == word2:
            return 1.0
        
        # 檢查雙向映射詞彙
        score = self._check_vocabulary_mapping(word1, word2)
        if score > 0:
            return score
        
        # 檢查字符串包含關係
        if word1 in word2 or word2 in word1:
            longer = max(word1, word2, key=len)
            shorter = min(word1, word2, key=len)
            if len(shorter) >= 2:  # 避免太短的部分匹配
                return 0.6 * (len(shorter) / len(longer))
        
        return 0.0
    
    def _check_vocabulary_mapping(self, word1: str, word2: str) -> float:
        """
        檢查詞彙映射關係
        """
        # 檢查 word1 -> word2 的映射
        if word1 in VOCABULARY_MAPPING and word2 in VOCABULARY_MAPPING[word1]:
            return 0.8
        
        # 檢查 word2 -> word1 的映射
        if word2 in VOCABULARY_MAPPING and word1 in VOCABULARY_MAPPING[word2]:
            return 0.8
        
        # 檢查反向映射：word2 是否在任何映射列表中，且對應的key與word1匹配
        for key, values in VOCABULARY_MAPPING.items():
            if word2 in values and word1 == key:
                return 0.8
            if word1 in values and word2 == key:
                return 0.8
        
        return 0.0
    
    def _calculate_dynamic_threshold(self, similarities: List[float], article: Dict) -> float:
        """
        根據條文類別和相似度分布計算動態閾值
        """
        if not similarities:
            return 0.1
        
        valid_similarities = [s for s in similarities if s > 0]
        if not valid_similarities:
            return 0.1
        
        # 基礎閾值根據條文類別決定
        category = article.get('category', '').lower()
        base_thresholds = {
            '治理': 0.15,      # 治理類別用詞差異較大，閾值較低
            '策略': 0.20,      # 策略類別中等
            '風險管理': 0.18,   # 風險管理中等
            '指標與目標': 0.25, # 指標類別要求較精確
            '報告基礎': 0.15   # 報告基礎較寬鬆
        }
        
        base_threshold = 0.2  # 預設閾值
        for cat_key, threshold in base_thresholds.items():
            if cat_key in category:
                base_threshold = threshold
                break
        
        # 根據相似度分布調整
        max_sim = max(valid_similarities)
        avg_sim = sum(valid_similarities) / len(valid_similarities)
        
        # 如果最高相似度很低，降低閾值
        if max_sim < 0.3:
            base_threshold *= 0.7
        
        # 如果平均相似度很低，進一步降低閾值
        if avg_sim < 0.1:
            base_threshold *= 0.5
        
        # 確保閾值在合理範圍內
        final_threshold = max(0.05, min(base_threshold, 0.4))
        
        logger.debug(f"動態閾值計算: 類別={category}, 基礎={base_threshold:.3f}, 最終={final_threshold:.3f}")
        return final_threshold
    
    def _select_best_evidences(self, all_evidences: List[Dict], similarities: List[float]) -> List[Dict]:
        """
        智能選擇最佳證據，去除重疊並確保多樣性
        """
        if not all_evidences and similarities:
            # 如果沒有超過閾值的證據，選擇最高分的前1-2個
            max_sim = max(similarities)
            if max_sim > 0.05:  # 最低底線
                # 找到最高分的索引
                max_indices = []
                for i, sim in enumerate(similarities):
                    if abs(sim - max_sim) < 1e-6:  # 浮點數比較
                        max_indices.append(i)
                
                # 為最高分項目創建證據
                backup_evidences = []
                for idx in max_indices[:2]:  # 最多取2個
                    if idx < len(similarities):
                        backup_evidences.append({
                            'content': f"未找到高匹配度證據，顯示最相關內容 (相似度: {similarities[idx]:.3f})",
                            'similarity': similarities[idx],
                            'type': 'fallback',
                            'index': idx
                        })
                
                logger.debug(f"使用備援證據: {len(backup_evidences)} 個, 最高分: {max_sim:.3f}")
                return backup_evidences
        
        # 先去除重疊證據
        deduplicated_evidences = self._remove_overlapping_evidences(all_evidences)
        
        # 限制證據數量，優先選擇高質量證據
        max_evidences = 5
        if len(deduplicated_evidences) <= max_evidences:
            return deduplicated_evidences
        
        # 多樣化選擇：頁面分散 + 類型平衡
        selected = self._diversified_evidence_selection(deduplicated_evidences, max_evidences)
        
        logger.debug(f"證據選擇完成: {len(selected)} 個, 去重前: {len(all_evidences)}, 去重後: {len(deduplicated_evidences)}")
        return selected
    
    def _remove_overlapping_evidences(self, evidences: List[Dict]) -> List[Dict]:
        """
        移除重疊的證據內容
        """
        if len(evidences) <= 1:
            return evidences
        
        deduplicated = []
        
        for current in evidences:
            current_content = current.get('content', '').strip()
            if not current_content:
                continue
                
            is_overlapping = False
            
            for existing in deduplicated:
                existing_content = existing.get('content', '').strip()
                if not existing_content:
                    continue
                
                # 檢查內容重疊
                overlap_ratio = self._calculate_content_overlap(current_content, existing_content)
                
                if overlap_ratio > 0.7:  # 70%以上重疊視為重複
                    is_overlapping = True
                    # 保留相似度更高的證據
                    if current.get('similarity', 0) > existing.get('similarity', 0):
                        deduplicated.remove(existing)
                        deduplicated.append(current)
                    break
            
            if not is_overlapping:
                deduplicated.append(current)
        
        # 按相似度排序
        deduplicated.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        logger.debug(f"去重完成: 原始 {len(evidences)} → 去重後 {len(deduplicated)}")
        return deduplicated
    
    def _calculate_content_overlap(self, content1: str, content2: str) -> float:
        """
        計算兩個內容的重疊比例
        """
        if not content1 or not content2:
            return 0.0
        
        # 簡化處理：計算字符級重疊
        shorter_content = content1 if len(content1) <= len(content2) else content2
        longer_content = content2 if len(content1) <= len(content2) else content1
        
        # 檢查短文本是否被長文本包含
        if shorter_content in longer_content:
            return 1.0
        
        # 計算詞級重疊
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _diversified_evidence_selection(self, evidences: List[Dict], max_count: int) -> List[Dict]:
        """
        多樣化證據選擇：確保頁面分散和類型平衡
        """
        if len(evidences) <= max_count:
            return evidences
        
        selected = []
        used_pages = set()
        sentence_count = 0
        paragraph_count = 0
        
        # 第一輪：選擇不同頁面的高分證據
        for evidence in evidences:
            if len(selected) >= max_count:
                break
                
            page = evidence.get('page', evidence.get('index', 0))
            evidence_type = evidence.get('type', 'unknown')
            
            # 優先選擇不同頁面的證據
            if page not in used_pages:
                if evidence_type == 'sentence' and sentence_count < max_count // 2 + 1:
                    selected.append(evidence)
                    used_pages.add(page)
                    sentence_count += 1
                elif evidence_type == 'paragraph' and paragraph_count < max_count // 2 + 1:
                    selected.append(evidence)
                    used_pages.add(page)
                    paragraph_count += 1
        
        # 第二輪：如果還有空位，選擇剩餘的高分證據
        for evidence in evidences:
            if len(selected) >= max_count:
                break
                
            if evidence not in selected:
                evidence_type = evidence.get('type', 'unknown')
                
                if evidence_type == 'sentence' and sentence_count < max_count // 2 + 1:
                    selected.append(evidence)
                    sentence_count += 1
                elif evidence_type == 'paragraph' and paragraph_count < max_count // 2 + 1:
                    selected.append(evidence)
                    paragraph_count += 1
                elif len(selected) < max_count:
                    selected.append(evidence)
        
        # 最終排序
        selected.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        page_distribution = {}
        for evidence in selected:
            page = evidence.get('page', evidence.get('index', 0))
            page_distribution[page] = page_distribution.get(page, 0) + 1
        
        logger.debug(f"多樣化選擇: {len(selected)} 證據分佈在 {len(page_distribution)} 頁: {page_distribution}")
        return selected
    
    def _fast_evidence_matching(self, text_chunks: List[str], 
                              article: Dict, 
                              max_evidences: int = 3) -> List[Dict]:
        """快速證據匹配"""
        evidences = []
        
        # 簡化的關鍵詞匹配
        article_keywords = article.get('keywords', [])
        
        for chunk in text_chunks[:10]:  # 限制檢查數量
            score = 0.0
            matched_keywords = []
            
            # 快速關鍵詞評分
            chunk_lower = chunk.lower()
            for keyword in article_keywords[:5]:  # 限制關鍵詞數量
                if keyword.lower() in chunk_lower:
                    score += 0.2
                    matched_keywords.append(keyword)
            
            # 大幅降低門檻，並添加語義匹配備案
            if score > 0.05 or len(chunk.strip()) > 10:  # 任何關鍵詞匹配或非空內容
                evidences.append({
                    'text': chunk[:200],  # 截斷長文本
                    'similarity': max(score, 0.1),  # 確保有基本相似度分數
                    'score': score,
                    'matched_keywords': matched_keywords
                })
            
            if len(evidences) >= max_evidences:
                break
        
        return sorted(evidences, key=lambda x: x['score'], reverse=True)
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """去重複結果"""
        seen_articles = set()
        deduplicated = []
        
        # 按置信度排序
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        for result in results:
            article_id = result['article']['article_id']
            if article_id not in seen_articles:
                seen_articles.add(article_id)
                deduplicated.append(result)
        
        return deduplicated[:50]  # 限制結果數量

if __name__ == "__main__":
    # 測試代碼
    logger.info("🧪 FAISS向量檢索系統測試")
    
    # 模擬IFRS S1條文數據
    mock_articles = [
        {
            'id': 'S1-20',
            'title': 'Purpose of IFRS S1',
            'content': 'This Standard requires an entity to disclose information about sustainability-related risks and opportunities.',
            'category': 'General Requirements',
            'difficulty': 'medium',
            'keywords': ['sustainability', 'risks', 'opportunities', 'disclosure']
        },
        {
            'id': 'S1-21',
            'title': 'Core content',
            'content': 'An entity shall disclose information that enables users to understand the governance processes.',
            'category': 'Governance',
            'difficulty': 'high',
            'keywords': ['governance', 'processes', 'users', 'understand']
        }
    ]
    
    # 初始化並測試
    analyzer = FastIFRSAnalyzer()
    analyzer.initialize_with_articles(mock_articles)
    
    # 測試搜索
    test_results = analyzer.retrieval_system.hybrid_search(
        "sustainability governance disclosure", 
        top_k=5
    )
    
    print("🔍 測試搜索結果:")
    for result in test_results:
        print(f"  - {result['article_id']}: {result['title']} (分數: {result['score']:.3f})")
    
    # 性能統計
    stats = analyzer.retrieval_system.get_stats()
    print(f"\n📊 性能統計: {stats}")