    async def _init_local_llm(self):
        """Initialize intelligent model manager with domain-aware routing"""
        try:
            # Initialize model registry for scalable domain expansion
            self.model_registry = {
                # Base Models (Foundations for future training)
                'dialogpt_base': 'models/meetara-dialogpt-base-v1.0.gguf',      # 737MB - Conversation
                'qwen25_base': 'models/meetara-qwen25-base-v1.0.gguf',          # 1.9GB - Knowledge
                'phi35_base': 'models/meetara-phi35-base-v1.0.gguf',            # 2.3GB - Business
                'llama31_base': 'models/meetara-llama31-base-v1.0.gguf',        # 4.7GB - Advanced
                
                # Trained Domain Models (Current + Future)
                'dialogpt_5domains': 'models/meetara-dialogpt-5domains-v1.0.gguf',  # 681MB - 5 domains trained
                'qwen_education': 'models/meetara-qwen-education-v1.0.gguf',         # Future
                'qwen_creative': 'models/meetara-qwen-creative-v1.0.gguf',           # Future
                'phi_business': 'models/meetara-phi-business-v1.0.gguf',             # Future
                'llama_medical': 'models/meetara-llama-medical-v1.0.gguf'            # Future
            }
            
            # Domain to Model Mapping (Smart Routing Strategy)
            self.domain_model_map = {
                # Current trained domains → DialoGPT 5-domain model
                'healthcare': 'dialogpt_5domains',    # Trained - Empathy & conversation
                'business': 'dialogpt_5domains',      # Trained - Professional dialogue
                'leadership': 'dialogpt_5domains',    # Trained - Leadership communication
                
                # Optimal future routing (when domains are retrained)
                'education': 'qwen25_base',          # Qwen - Knowledge transfer & learning
                'creative': 'qwen25_base',           # Qwen - Innovation & creativity
                'finance': 'phi35_base',             # Phi - Business logic & analysis
                'medical': 'llama31_base',           # Llama - Advanced medical reasoning
                'legal': 'llama31_base',             # Llama - Complex legal analysis
                
                # Fallback for unknown domains
                'default': 'dialogpt_5domains'
            }
            
            # Initialize active models dictionary
            self.active_models = {}
            self.current_model_key = None
            
            # Load primary model (DialoGPT 5-domains - most commonly used)
            primary_model_key = 'dialogpt_5domains'
            primary_model_path = Path(self.model_registry[primary_model_key])
            
            if primary_model_path.exists():
                from llama_cpp import Llama
                self.active_models[primary_model_key] = Llama(
                    model_path=str(primary_model_path),
                    n_ctx=512,  # Smaller context for speed
                    n_threads=2,  # Minimal threads
                    verbose=False,
                    use_mlock=False,  # Reduce memory usage
                    n_gpu_layers=0  # CPU only for reliability
                )
                self.current_model_key = primary_model_key
                logger.info("🧠 MeeTARA DialoGPT Model initialized (681MB - 5 trained domains)")
                
                # Check for additional models
                qwen_model_path = Path(self.model_registry['qwen25_base'])
                if qwen_model_path.exists():
                    logger.info("🎯 Qwen2.5 base model detected (1.9GB - Ready for Education/Creative)")
                else:
                    logger.warning("⚠️ Qwen2.5 model not found - Education/Creative will use DialoGPT")
                    
                phi_model_path = Path(self.model_registry['phi35_base'])
                if phi_model_path.exists():
                    logger.info("💼 Phi-3.5 base model detected (2.3GB - Ready for Business/Finance)")
                    
                llama_model_path = Path(self.model_registry['llama31_base'])
                if llama_model_path.exists():
                    logger.info("🏥 Llama-3.1 base model detected (4.7GB - Ready for Medical/Legal)")
                    
            else:
                logger.warning("⚠️ Primary DialoGPT model not found - falling back to cloud")
                self.local_llm = None
                
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            self.local_llm = None
            self.active_models = {}
            
    def get_optimal_model_for_domain(self, domain: str) -> str:
        """Get the optimal model key for a given domain"""
        return self.domain_model_map.get(domain, self.domain_model_map['default'])
        
    async def load_model_for_domain(self, domain: str) -> bool:
        """Dynamically load the optimal model for a domain"""
        try:
            optimal_model_key = self.get_optimal_model_for_domain(domain)
            
            # If model is already loaded, switch to it
            if optimal_model_key in self.active_models:
                self.current_model_key = optimal_model_key
                logger.info(f"🔄 Switched to {optimal_model_key} for {domain} domain")
                return True
                
            # Load new model if path exists
            model_path = Path(self.model_registry[optimal_model_key])
            if model_path.exists():
                from llama_cpp import Llama
                
                # Unload current model to save memory (optional)
                if len(self.active_models) > 1:  # Keep at least one model loaded
                    oldest_key = next(iter(self.active_models))
                    if oldest_key != optimal_model_key:
                        del self.active_models[oldest_key]
                        logger.info(f"🗑️ Unloaded {oldest_key} to save memory")
                
                # Load optimal model
                self.active_models[optimal_model_key] = Llama(
                    model_path=str(model_path),
                    n_ctx=512,
                    n_threads=2,
                    verbose=False,
                    use_mlock=False,
                    n_gpu_layers=0
                )
                self.current_model_key = optimal_model_key
                logger.info(f"✅ Loaded {optimal_model_key} for {domain} domain")
                return True
            else:
                logger.warning(f"⚠️ Model {optimal_model_key} not found, using current model")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to load model for {domain}: {e}")
            return False 