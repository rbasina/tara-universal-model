# TARA Universal Model - Model Naming Strategy

**📅 Last Updated**: June 26, 2025  
**🎯 Purpose**: Document model naming conventions and mapping strategy  
**🔄 Status**: Active Reference Document  
**🚀 Architecture**: MeeTARA Trinity Architecture

---

## 🎯 **Executive Summary**

This document outlines the model naming strategy used in the TARA Universal Model project. We use generic, functional names in our public-facing documentation and dashboards while maintaining precise mappings to the actual base models in our configuration files.

---

## 📋 **Model Naming Convention**

### **Generic Naming Approach**

We use descriptive, function-based names that highlight the model's strengths and parameter count:

| Generic Name | Purpose | Parameters |
|--------------|---------|------------|
| **Premium-8B-Instruct** | Complex reasoning, strategic thinking | 8B |
| **Technical-3.8B-Instruct** | Technical knowledge, creative problem-solving | 3.8B |
| **Efficient-1B-Instruct** | Fast, efficient daily interactions | 1B |
| **DialoGPT-medium** | Therapeutic communication, empathy | 345M |

### **Naming Pattern**

Our generic names follow this pattern:
```
[Function]-[Parameter Count]-[Type]
```

Where:
- **Function**: Primary strength (Premium, Technical, Efficient)
- **Parameter Count**: Size in billions/millions of parameters
- **Type**: Model type (Instruct, Base, etc.)

---

## 🔄 **Model Mapping System**

### **Mapping Configuration Files**

We maintain two key mapping files:

1. **`configs/model_mapping.json`**
   - Development reference mapping
   - Links generic names to actual models
   - Includes parameter counts and domain assignments

2. **`configs/model_mapping_production.json`**
   - Production deployment mapping
   - Includes detailed metadata for deployment
   - Maps domains to appropriate models
   - Tracks current vs. optimal model assignments

### **Mapping Structure**

```json
{
  "generic_to_actual": {
    "Premium-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Technical-3.8B-Instruct": "microsoft/Phi-3.5-mini-instruct",
    "Efficient-1B-Instruct": "meta-llama/Llama-3.2-1B-instruct",
    "DialoGPT-medium": "microsoft/DialoGPT-medium"
  }
}
```

---

## 🎯 **Domain-Model Assignments**

### **Optimal Model Selection**

Each domain is assigned to its optimal model based on the model's strengths:

| Domain | Optimal Model | Rationale |
|--------|---------------|-----------|
| Healthcare | DialoGPT-medium | Therapeutic communication, empathy |
| Business | Premium-8B-Instruct | Strategic thinking, complex reasoning |
| Education | Technical-3.8B-Instruct | Technical knowledge, instruction following |
| Creative | Technical-3.8B-Instruct | Creative problem-solving |
| Leadership | Premium-8B-Instruct | Strategic planning, team management |

### **Current vs. Optimal Assignments**

The `current_phase_mappings` section in `model_mapping_production.json` tracks:
- Current model being used for each domain
- Optimal model for each domain
- Training status
- Whether improvement is needed
- Potential improvement percentage

---

## 🔧 **Implementation Details**

### **Code Integration**

Our code uses the mapping files to translate between generic and actual model names:

```python
def get_actual_model_name(generic_name):
    """Convert generic model name to actual model name."""
    with open("configs/model_mapping.json", "r") as f:
        mappings = json.load(f)
    return mappings["generic_to_actual"].get(generic_name)

def get_optimal_model_for_domain(domain):
    """Get optimal model for a specific domain."""
    with open("configs/model_mapping_production.json", "r") as f:
        mappings = json.load(f)
    return mappings["domain_to_model"].get(domain)
```

### **Dashboard Integration**

The domain optimization dashboard uses generic names while the backend systems reference the actual model names through the mapping files.

---

## 📊 **Benefits of This Approach**

1. **Clarity**: Descriptive names clearly communicate model functions
2. **Flexibility**: Easy to swap underlying models without changing documentation
3. **Future-Proofing**: Can incorporate new models under the same functional categories
4. **Legal Compliance**: Avoids potential trademark issues in public-facing materials
5. **Consistency**: Maintains consistent naming across documentation and UI

---

## 🚀 **Conclusion**

This model naming strategy enables us to:
- Clearly communicate model functions and strengths
- Maintain precise technical references to actual models
- Track optimal model assignments for each domain
- Support future model upgrades and replacements

By maintaining this dual naming system with proper mapping, we ensure both clarity in communication and technical precision in implementation. 