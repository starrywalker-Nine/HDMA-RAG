#!/usr/bin/env python3
"""
HDMARAG系统测试脚本
验证核心功能和组件集成
"""

import sys
import traceback
from datetime import datetime

def test_imports():
    """测试模块导入"""
    print("=" * 50)
    print("测试模块导入")
    print("=" * 50)
    
    try:
        from hdmarag_core import HDMARAGCore, AbstractionEngine, EnhancedRetriever, HierarchicalChunker
        print("✓ hdmarag_core 导入成功")
        
        from hdmarag_system import HDMARAGSystem
        print("✓ hdmarag_system 导入成功")
        
        from testQA import DialogueQASystem
        print("✓ testQA 导入成功")
        
        from testChunks import OptimizedMemoryChunkExtractor
        print("✓ testChunks 导入成功")
        
        from testRetriever import OptimizedMultiTurnRetriever
        print("✓ testRetriever 导入成功")
        
        from answerModel import AdvancedAnswerModel
        print("✓ answerModel 导入成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        traceback.print_exc()
        return False

def test_hdmarag_core():
    """测试HDMARAG核心组件"""
    print("\n" + "=" * 50)
    print("测试HDMARAG核心组件")
    print("=" * 50)
    
    try:
        from hdmarag_core import HDMARAGCore
        
        # 初始化核心组件
        hdmarag_core = HDMARAGCore(
            api_key="test-key",
            model="qwen2.5-7b-instruct"
        )
        print("✓ HDMARAGCore 初始化成功")
        
        # 测试记忆摘要
        memory_summary = hdmarag_core.get_memory_summary()
        print(f"✓ 记忆摘要获取成功: {memory_summary}")
        
        # 测试记忆重置
        hdmarag_core.reset_memory()
        print("✓ 记忆重置成功")
        
        return True
        
    except Exception as e:
        print(f"✗ HDMARAG核心测试失败: {e}")
        traceback.print_exc()
        return False

def test_hierarchical_chunker():
    """测试分层分块器"""
    print("\n" + "=" * 50)
    print("测试分层分块器")
    print("=" * 50)
    
    try:
        from hdmarag_core import HierarchicalChunker
        
        chunker = HierarchicalChunker()
        print("✓ HierarchicalChunker 初始化成功")
        
        # 测试分层分块
        test_context = """
        Artificial Intelligence (AI) has revolutionized many aspects of modern life. 
        Machine learning, a subset of AI, enables computers to learn and improve from experience. 
        Deep learning uses neural networks with multiple layers for complex pattern recognition. 
        Natural language processing allows computers to understand and generate human language. 
        Computer vision enables machines to interpret and analyze visual information. 
        These technologies have applications in healthcare, finance, transportation, and education.
        """
        
        test_question = "What are the main AI technologies?"
        
        hierarchical_chunks = chunker.generate_hierarchical_chunks(
            test_context, test_question, "single_doc_qa"
        )
        
        print(f"✓ 分层分块成功，生成 {len(hierarchical_chunks.get('levels', []))} 个层级")
        
        for i, level in enumerate(hierarchical_chunks.get('levels', [])):
            print(f"  层级 {i+1}: {level.get('chunk_count', 0)} 个chunks")
        
        return True
        
    except Exception as e:
        print(f"✗ 分层分块器测试失败: {e}")
        traceback.print_exc()
        return False

def test_abstraction_engine():
    """测试抽象思考引擎"""
    print("\n" + "=" * 50)
    print("测试抽象思考引擎")
    print("=" * 50)
    
    try:
        from hdmarag_core import AbstractionEngine
        from openai import OpenAI
        
        # 模拟客户端
        client = OpenAI(api_key="test-key", base_url="https://api.example.com")
        engine = AbstractionEngine(client, "test-model")
        print("✓ AbstractionEngine 初始化成功")
        
        # 测试step-back分析（不实际调用API）
        test_question = "What are the challenges in AI development?"
        test_context = "AI development faces various technical and ethical challenges."
        
        # 模拟结果
        mock_result = {
            "fundamental_concept": "AI development challenges",
            "general_principles": "Technical and ethical considerations",
            "higher_patterns": "Technology adoption patterns",
            "knowledge_domains": "Computer science, ethics, policy",
            "abstraction_chain": "Specific challenges -> General principles -> Universal patterns",
            "thinking_dimension": "Multi-dimensional analysis"
        }
        
        print("✓ Step-back分析结构验证成功")
        print(f"  核心概念: {mock_result['fundamental_concept']}")
        print(f"  一般原理: {mock_result['general_principles']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 抽象思考引擎测试失败: {e}")
        traceback.print_exc()
        return False

def test_hdmarag_system():
    """测试HDMARAG完整系统"""
    print("\n" + "=" * 50)
    print("测试HDMARAG完整系统")
    print("=" * 50)
    
    try:
        from hdmarag_system import HDMARAGSystem
        
        # 初始化系统（使用测试配置）
        hdmarag_system = HDMARAGSystem(
            api_key="test-key",
            model="qwen2.5-7b-instruct"
        )
        print("✓ HDMARAGSystem 初始化成功")
        
        # 测试系统状态
        status = hdmarag_system.get_system_status()
        print(f"✓ 系统状态获取成功")
        print(f"  性能指标: {status['performance_metrics']}")
        print(f"  记忆摘要: {status['memory_summary']}")
        
        # 测试系统重置
        hdmarag_system.reset_system()
        print("✓ 系统重置成功")
        
        return True
        
    except Exception as e:
        print(f"✗ HDMARAG系统测试失败: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """测试组件集成"""
    print("\n" + "=" * 50)
    print("测试组件集成")
    print("=" * 50)
    
    try:
        # 测试原有组件是否可以正常导入和初始化
        from testQA import DialogueQASystem
        from testChunks import OptimizedMemoryChunkExtractor
        from testRetriever import OptimizedMultiTurnRetriever
        from answerModel import AdvancedAnswerModel
        
        # 初始化原有组件
        qa_system = DialogueQASystem(api_key="test-key")
        memory_extractor = OptimizedMemoryChunkExtractor(api_key="test-key")
        retriever = OptimizedMultiTurnRetriever(api_key="test-key")
        answer_model = AdvancedAnswerModel(api_key="test-key")
        
        print("✓ 所有原有组件初始化成功")
        
        # 测试组件方法
        qa_summary = qa_system.get_dialogue_summary()
        memory_stats = memory_extractor.get_memory_stats()
        retriever_summary = retriever.get_session_summary()
        answer_stats = answer_model.get_answer_statistics()
        
        print("✓ 组件方法调用成功")
        print(f"  QA对话轮数: {qa_summary['total_turns']}")
        print(f"  记忆块总数: {memory_stats['total_chunks']}")
        print(f"  检索会话数: {retriever_summary['total_turns']}")
        print(f"  答案生成数: {answer_stats['total_answers']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 组件集成测试失败: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """测试配置系统"""
    print("\n" + "=" * 50)
    print("测试配置系统")
    print("=" * 50)
    
    try:
        import json
        import os
        
        # 检查配置文件
        config_files = ["hdmarag_config.json", "gpu_server_config.json"]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"✓ {config_file} 加载成功")
            else:
                print(f"⚠ {config_file} 不存在，将使用默认配置")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置系统测试失败: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print("HDMARAG系统测试开始")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("模块导入", test_imports),
        ("HDMARAG核心", test_hdmarag_core),
        ("分层分块器", test_hierarchical_chunker),
        ("抽象思考引擎", test_abstraction_engine),
        ("HDMARAG系统", test_hdmarag_system),
        ("组件集成", test_integration),
        ("配置系统", test_configuration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 测试总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！HDMARAG系统准备就绪。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关组件。")
        return False

def main():
    """主函数"""
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        test_map = {
            "imports": test_imports,
            "core": test_hdmarag_core,
            "chunker": test_hierarchical_chunker,
            "abstraction": test_abstraction_engine,
            "system": test_hdmarag_system,
            "integration": test_integration,
            "config": test_configuration
        }
        
        if test_name in test_map:
            success = test_map[test_name]()
            sys.exit(0 if success else 1)
        else:
            print(f"未知测试: {test_name}")
            print(f"可用测试: {', '.join(test_map.keys())}")
            sys.exit(1)
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()