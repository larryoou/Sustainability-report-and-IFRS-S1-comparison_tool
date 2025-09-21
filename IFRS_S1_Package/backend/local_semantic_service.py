from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import uvicorn
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local Semantic Analysis Service", version="1.0.0")

# 允許CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EncodeRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = "primary"

class SimilarityRequest(BaseModel):
    source_embedding: List[float]
    candidate_embeddings: List[List[float]]

class SmartSimilarityRequest(BaseModel):
    query_text: str
    candidate_texts: List[str]
    threshold: Optional[float] = 0.3

class MultiLevelSimilarityRequest(BaseModel):
    query_title: Optional[str] = ""
    query_content: str
    candidate_sentences: List[str]
    candidate_paragraphs: List[str]
    weights: Optional[Dict[str, float]] = {
        "title_weight": 0.3,
        "content_weight": 0.4, 
        "sentence_weight": 0.15,
        "paragraph_weight": 0.15
    }
    threshold: Optional[float] = 0.3

class SemanticAnalysisService:
    def __init__(self, enable_gpu_optimization=True, mixed_precision=True):
        """
        初始化語義分析服務
        
        Args:
            enable_gpu_optimization: 是否啟用GPU優化
            mixed_precision: 是否使用混合精度加速
        """
        # GPU優化設置
        self.enable_gpu_optimization = enable_gpu_optimization and torch.cuda.is_available()
        self.mixed_precision = mixed_precision and self.enable_gpu_optimization
        self.device = 'cuda' if self.enable_gpu_optimization else 'cpu'
        
        # GPU記憶體優化
        if self.enable_gpu_optimization:
            torch.backends.cudnn.benchmark = True  # 優化CNN性能
            torch.backends.cuda.matmul.allow_tf32 = True  # 允許TF32加速
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
        
        logger.info(f"使用設備: {self.device}")
        logger.info(f"GPU優化: {self.enable_gpu_optimization}")
        logger.info(f"混合精度: {self.mixed_precision}")
        
        if self.enable_gpu_optimization:
            gpu_info = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {gpu_info.name}, 記憶體: {gpu_info.total_memory // 1024**3}GB")
        
        # 載入模型
        self.models = {}
        self.model_configs = {
            # 所有預設模型已移除 - 使用加速服務端點
        }
        
        self._load_models()
    
    def _load_models(self):
        """載入模型配置"""
        try:
            # 所有本地模型已移除，服務作為代理使用加速端點
            logger.info("本地模型服務已配置為代理模式")
            logger.info("將請求轉發至加速服務端點 localhost:8004")
            self.current_primary = 'proxy_mode'
            self.models = {}  # 清空本地模型
            
            # BGE備援模型已移除
            # logger.info("載入備援模型: BGE-large-zh-v1.5")
            # fallback_name = self.model_configs['bge-zh']['model_name']
            # self.models['fallback'] = SentenceTransformer(fallback_name, device=self.device)
            # self.current_fallback = 'bge-zh'
            
            # GPU記憶體優化
            if self.enable_gpu_optimization:
                for model_key, model in self.models.items():
                    model.half() if self.mixed_precision else model.float()
                    logger.info(f"模型 {model_key} 已優化為 {'FP16' if self.mixed_precision else 'FP32'}")
            
            logger.info(f"✅ 所有模型載入完成，主要模型: {self.current_primary}")
            
        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
            raise
    
    def get_optimal_batch_size(self, model_key: str, text_count: int) -> int:
        """根據GPU記憶體動態調整批次大小"""
        if not self.enable_gpu_optimization:
            return min(8, text_count)
        
        config = self.model_configs.get(self.current_primary if model_key == 'primary' else self.current_fallback, 
                                       self.model_configs['voyage-3-large'])  # 使用voyage作為默認
        base_batch_size = config['batch_size']
        
        # 根據GPU記憶體使用動態調整
        if torch.cuda.is_available():
            memory_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            memory_ratio = memory_free / torch.cuda.get_device_properties(0).total_memory
            
            if memory_ratio > 0.7:
                adjusted_batch_size = int(base_batch_size * 1.5)
            elif memory_ratio > 0.5:
                adjusted_batch_size = base_batch_size
            else:
                adjusted_batch_size = max(4, int(base_batch_size * 0.7))
                
            logger.debug(f"GPU記憶體使用率: {(1-memory_ratio)*100:.1f}%, 批次大小: {adjusted_batch_size}")
            return min(adjusted_batch_size, text_count)
        
        return min(base_batch_size, text_count)
    
    def encode_texts(self, texts: List[str], model_name: str = 'primary') -> List[List[float]]:
        """
        GPU優化的文本編碼，支援大批次處理
        對應你的 encode_texts 函數，但添加了GPU優化
        """
        try:
            model = self.models.get(model_name, self.models['primary'])
            
            # 動態批次大小
            batch_size = self.get_optimal_batch_size(model_name, len(texts))
            
            # GPU記憶體預熱
            if self.enable_gpu_optimization and len(texts) > batch_size:
                torch.cuda.empty_cache()
            
            # 分批編碼以節省記憶體並提升效率
            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                logger.debug(f"處理批次 {batch_num}/{total_batches}, 大小: {len(batch_texts)}")
                
                # 使用上下文管理器進行GPU優化
                with torch.cuda.amp.autocast() if self.mixed_precision else torch.no_grad():
                    batch_embeddings = model.encode(
                        batch_texts,
                        convert_to_tensor=True,
                        device=self.device,
                        show_progress_bar=False,
                        batch_size=len(batch_texts)  # 避免內部重複分批
                    )
                
                # 轉移至CPU以節省GPU記憶體
                all_embeddings.append(batch_embeddings.cpu())
                
                # 定期清理GPU記憶體
                if self.enable_gpu_optimization and batch_num % 4 == 0:
                    torch.cuda.empty_cache()
            
            # 合併所有批次
            final_embeddings = torch.cat(all_embeddings, dim=0)
            
            logger.info(f"✅ 編碼完成: {len(texts)} 個文本, 使用模型: {self.current_primary if model_name == 'primary' else self.current_fallback}")
            
            return final_embeddings.numpy().tolist()
        
        except Exception as e:
            logger.error(f"編碼失敗: {e}")
            if model_name != 'fallback':
                logger.info("嘗試使用備援模型")
                return self.encode_texts(texts, 'fallback')
            raise
    
    def get_model_info(self) -> Dict:
        """獲取當前模型配置信息"""
        return {
            'primary_model': self.current_primary,
            'fallback_model': self.current_fallback,
            'device': self.device,
            'gpu_optimization': self.enable_gpu_optimization,
            'mixed_precision': self.mixed_precision,
            'model_configs': self.model_configs
        }
    
    def calculate_similarity(self, query_emb: List[float], 
                           corpus_emb: List[List[float]]) -> List[float]:
        """計算cosine相似度，對應你的 calculate_similarity 函數"""
        try:
            query_tensor = torch.tensor([query_emb], device=self.device)
            corpus_tensor = torch.tensor(corpus_emb, device=self.device)
            
            # 使用 util.pytorch_cos_sim 計算cosine相似度
            cos_scores = util.pytorch_cos_sim(query_tensor, corpus_tensor)[0]
            
            return cos_scores.cpu().numpy().tolist()
        
        except Exception as e:
            logger.error(f"相似度計算失敗: {e}")
            raise
    
    def calculate_similarity_with_fallback(self, query_text: str, 
                                         candidate_texts: List[str],
                                         threshold: float = 0.3) -> dict:
        """智能相似度計算，整合你的備援邏輯"""
        try:
            # 編碼查詢文本和候選文本
            all_texts = [query_text] + candidate_texts
            
            # 使用主要模型編碼
            logger.info("使用主要模型進行編碼...")
            primary_embeddings = self.encode_texts(all_texts, 'primary')
            
            query_embedding = primary_embeddings[0]
            corpus_embeddings = primary_embeddings[1:]
            
            # 計算主要模型的相似度
            primary_scores = self.calculate_similarity(query_embedding, corpus_embeddings)
            max_primary_score = max(primary_scores) if primary_scores else 0
            
            logger.info(f"主要模型最高相似度: {max_primary_score:.4f}, 閾值: {threshold}")
            
            # 檢查是否需要使用備援模型
            if max_primary_score < threshold:
                logger.info(f"主要模型分數 ({max_primary_score:.4f}) 低於閾值 ({threshold})，啟用備援模型")
                
                # 使用備援模型編碼
                fallback_embeddings = self.encode_texts(all_texts, 'fallback')
                fallback_query_embedding = fallback_embeddings[0]
                fallback_corpus_embeddings = fallback_embeddings[1:]
                
                # 計算備援模型的相似度
                fallback_scores = self.calculate_similarity(fallback_query_embedding, fallback_corpus_embeddings)
                
                # 取較大值進行融合
                final_scores = []
                for i in range(len(primary_scores)):
                    final_score = max(primary_scores[i], fallback_scores[i])
                    final_scores.append(final_score)
                
                max_fallback_score = max(fallback_scores) if fallback_scores else 0
                max_final_score = max(final_scores) if final_scores else 0
                
                logger.info(f"備援模型最高相似度: {max_fallback_score:.4f}")
                logger.info(f"融合後最高相似度: {max_final_score:.4f}")
                
                return {
                    'scores': final_scores,
                    'primary_scores': primary_scores,
                    'fallback_scores': fallback_scores,
                    'max_primary': max_primary_score,
                    'max_fallback': max_fallback_score,
                    'max_final': max_final_score,
                    'used_fallback': True,
                    'threshold': threshold
                }
            else:
                logger.info("主要模型分數滿足閾值，僅使用主要模型結果")
                return {
                    'scores': primary_scores,
                    'primary_scores': primary_scores,
                    'fallback_scores': None,
                    'max_primary': max_primary_score,
                    'max_fallback': None,
                    'max_final': max_primary_score,
                    'used_fallback': False,
                    'threshold': threshold
                }
                
        except Exception as e:
            logger.error(f"智能相似度計算失敗: {e}")
            raise
    
    def calculate_multi_level_similarity(self, query_title: str, query_content: str,
                                       candidate_sentences: List[str], 
                                       candidate_paragraphs: List[str],
                                       weights: Dict[str, float],
                                       threshold: float = 0.3) -> dict:
        """多層級相似度整合計算"""
        try:
            logger.info("開始多層級相似度計算...")
            
            # 準備查詢文本
            query_parts = []
            query_labels = []
            
            if query_title and query_title.strip():
                query_parts.append(query_title.strip())
                query_labels.append("title")
            
            if query_content and query_content.strip():
                query_parts.append(query_content.strip())
                query_labels.append("content")
            
            # 組合查詢（用於整體相似度）
            combined_query = " ".join(query_parts)
            
            results = {}
            
            # 1. 計算標題和內容與句子的相似度
            if candidate_sentences:
                logger.info(f"計算與 {len(candidate_sentences)} 個句子的相似度...")
                sentence_results = {}
                
                for i, (query_part, label) in enumerate(zip(query_parts, query_labels)):
                    sim_result = self.calculate_similarity_with_fallback(
                        query_part, candidate_sentences, threshold
                    )
                    sentence_results[label] = sim_result
                
                # 整體查詢與句子相似度
                combined_sentence_result = self.calculate_similarity_with_fallback(
                    combined_query, candidate_sentences, threshold
                )
                
                results['sentence_similarities'] = {
                    'by_component': sentence_results,
                    'combined': combined_sentence_result
                }
            
            # 2. 計算標題和內容與段落的相似度
            if candidate_paragraphs:
                logger.info(f"計算與 {len(candidate_paragraphs)} 個段落的相似度...")
                paragraph_results = {}
                
                for i, (query_part, label) in enumerate(zip(query_parts, query_labels)):
                    sim_result = self.calculate_similarity_with_fallback(
                        query_part, candidate_paragraphs, threshold
                    )
                    paragraph_results[label] = sim_result
                
                # 整體查詢與段落相似度
                combined_paragraph_result = self.calculate_similarity_with_fallback(
                    combined_query, candidate_paragraphs, threshold
                )
                
                results['paragraph_similarities'] = {
                    'by_component': paragraph_results,
                    'combined': combined_paragraph_result
                }
            
            # 3. 加權整合分數
            weighted_scores = self._calculate_weighted_scores(results, weights, query_labels)
            results['weighted_scores'] = weighted_scores
            
            # 4. 分析結果
            analysis = self._analyze_multi_level_results(results, weights, threshold)
            results['analysis'] = analysis
            
            logger.info(f"多層級相似度計算完成，最高加權分數: {analysis['max_weighted_score']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"多層級相似度計算失敗: {e}")
            raise
    
    def _calculate_weighted_scores(self, results: dict, weights: Dict[str, float], 
                                 query_labels: List[str]) -> dict:
        """計算加權分數"""
        weighted_scores = {
            'sentence_level': [],
            'paragraph_level': [],
            'combined': []
        }
        
        # 句子層級加權
        if 'sentence_similarities' in results:
            sentence_scores = []
            
            # 按組件加權
            component_scores = []
            for label in query_labels:
                if label in results['sentence_similarities']['by_component']:
                    component_result = results['sentence_similarities']['by_component'][label]
                    weight_key = f"{label}_weight"
                    weight = weights.get(weight_key, 0.0)
                    
                    weighted = [score * weight for score in component_result['scores']]
                    component_scores.append(weighted)
            
            # 組合各組件分數
            if component_scores:
                num_candidates = len(component_scores[0])
                for i in range(num_candidates):
                    combined_score = sum(scores[i] for scores in component_scores)
                    sentence_scores.append(combined_score)
            
            # 添加句子層級權重
            sentence_weight = weights.get('sentence_weight', 0.15)
            weighted_scores['sentence_level'] = [score * sentence_weight for score in sentence_scores]
        
        # 段落層級加權（類似處理）
        if 'paragraph_similarities' in results:
            paragraph_scores = []
            
            component_scores = []
            for label in query_labels:
                if label in results['paragraph_similarities']['by_component']:
                    component_result = results['paragraph_similarities']['by_component'][label]
                    weight_key = f"{label}_weight"
                    weight = weights.get(weight_key, 0.0)
                    
                    weighted = [score * weight for score in component_result['scores']]
                    component_scores.append(weighted)
            
            if component_scores:
                num_candidates = len(component_scores[0])
                for i in range(num_candidates):
                    combined_score = sum(scores[i] for scores in component_scores)
                    paragraph_scores.append(combined_score)
            
            paragraph_weight = weights.get('paragraph_weight', 0.15)
            weighted_scores['paragraph_level'] = [score * paragraph_weight for score in paragraph_scores]
        
        # 最終組合分數
        max_candidates = max(
            len(weighted_scores.get('sentence_level', [])),
            len(weighted_scores.get('paragraph_level', []))
        )
        
        for i in range(max_candidates):
            sentence_score = weighted_scores['sentence_level'][i] if i < len(weighted_scores.get('sentence_level', [])) else 0
            paragraph_score = weighted_scores['paragraph_level'][i] if i < len(weighted_scores.get('paragraph_level', [])) else 0
            
            combined_score = sentence_score + paragraph_score
            weighted_scores['combined'].append(combined_score)
        
        return weighted_scores
    
    def _analyze_multi_level_results(self, results: dict, weights: Dict[str, float], 
                                   threshold: float) -> dict:
        """分析多層級結果"""
        analysis = {
            'max_weighted_score': 0,
            'sentence_analysis': {},
            'paragraph_analysis': {},
            'weight_distribution': weights,
            'threshold': threshold
        }
        
        # 分析加權分數
        if 'weighted_scores' in results:
            weighted = results['weighted_scores']
            
            if weighted.get('combined'):
                analysis['max_weighted_score'] = max(weighted['combined'])
                analysis['avg_weighted_score'] = sum(weighted['combined']) / len(weighted['combined'])
            
            if weighted.get('sentence_level'):
                analysis['sentence_analysis'] = {
                    'max_score': max(weighted['sentence_level']),
                    'avg_score': sum(weighted['sentence_level']) / len(weighted['sentence_level']),
                    'count': len(weighted['sentence_level'])
                }
            
            if weighted.get('paragraph_level'):
                analysis['paragraph_analysis'] = {
                    'max_score': max(weighted['paragraph_level']),
                    'avg_score': sum(weighted['paragraph_level']) / len(weighted['paragraph_level']),
                    'count': len(weighted['paragraph_level'])
                }
        
        return analysis

# 初始化服務
service = SemanticAnalysisService()

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {"status": "healthy", "device": service.device, "models_loaded": len(service.models)}

@app.post("/encode")
async def encode_texts_endpoint(request: EncodeRequest):
    """文本編碼端點"""
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="文本列表不能為空")
        
        embeddings = service.encode_texts(request.texts, request.model)
        
        return {
            "embeddings": embeddings,
            "model_used": request.model,
            "count": len(embeddings)
        }
    
    except Exception as e:
        logger.error(f"編碼請求失敗: {e}")
        raise HTTPException(status_code=500, detail=f"編碼失敗: {str(e)}")

@app.post("/similarity")
async def calculate_similarity_endpoint(request: SimilarityRequest):
    """相似度計算端點"""
    try:
        similarities = service.calculate_similarity(
            request.source_embedding,
            request.candidate_embeddings
        )
        
        return {
            "similarities": similarities,
            "count": len(similarities)
        }
    
    except Exception as e:
        logger.error(f"相似度計算失敗: {e}")
        raise HTTPException(status_code=500, detail=f"相似度計算失敗: {str(e)}")

@app.post("/smart_similarity")
async def smart_similarity_endpoint(request: SmartSimilarityRequest):
    """智能相似度計算端點，整合備援模型邏輯"""
    try:
        result = service.calculate_similarity_with_fallback(
            request.query_text,
            request.candidate_texts,
            request.threshold
        )
        
        return result
    
    except Exception as e:
        logger.error(f"智能相似度計算失敗: {e}")
        raise HTTPException(status_code=500, detail=f"智能相似度計算失敗: {str(e)}")

@app.post("/multi_level_similarity")
async def multi_level_similarity_endpoint(request: MultiLevelSimilarityRequest):
    """多層級相似度計算端點，支援句子和段落加權整合"""
    try:
        result = service.calculate_multi_level_similarity(
            request.query_title,
            request.query_content,
            request.candidate_sentences,
            request.candidate_paragraphs,
            request.weights,
            request.threshold
        )
        
        return result
    
    except Exception as e:
        logger.error(f"多層級相似度計算失敗: {e}")
        raise HTTPException(status_code=500, detail=f"多層級相似度計算失敗: {str(e)}")

@app.post("/calculate_similarity/")
async def legacy_similarity_endpoint(request: dict):
    """兼容原有API格式的相似度端點，支援智能備援"""
    try:
        source = request.get('source', '')
        candidates = request.get('candidates', [])
        threshold = request.get('threshold', 0.3)  # 支援自定義閾值
        
        if not source or not candidates:
            raise HTTPException(status_code=400, detail="源文本和候選文本不能為空")
        
        # 使用智能備援功能
        result = service.calculate_similarity_with_fallback(source, candidates, threshold)
        
        # 返回兼容格式，同時提供詳細信息
        return {
            "scores": result['scores'],
            "details": {
                "max_score": result['max_final'],
                "used_fallback": result['used_fallback'],
                "threshold": result['threshold'],
                "primary_max": result['max_primary'],
                "fallback_max": result.get('max_fallback')
            }
        }
    
    except Exception as e:
        logger.error(f"Legacy API 請求失敗: {e}")
        raise HTTPException(status_code=500, detail=f"計算失敗: {str(e)}")

if __name__ == "__main__":
    # 啟動服務
    uvicorn.run(
        "local_semantic_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )