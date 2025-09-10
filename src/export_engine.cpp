#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <chrono>
#include <unistd.h>
#include <cstring>
#include "efficientIdxNMSPlugin.h"
#include "common.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;
using namespace nvinfer1;

static Logger gLogger(ILogger::Severity::kERROR);

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("./export model.onnx model.engine fp16\n");
        return 0;
    }
    std::string onnxFile(argv[1]);
    std::string trtFile(argv[2]);
    std::string FP_mode(argv[3]);

    CHECK_CUDA(cudaSetDevice(0));
    ICudaEngine *engine = nullptr;

    std::vector<Dims32> min_shapes = {{1, {1, 3, 640, 640}}};
    std::vector<Dims32> opt_shapes = {{1, {1, 3, 640, 640}}};
    std::vector<Dims32> max_shapes = {{1, {1, 3, 640, 640}}};
    Dims4 min_dims(1, 3, 640, 640);

    if (access(trtFile.c_str(), F_OK) == 0)
    {
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0)
        {
            std::cout << "Failed getting serialized engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime{createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr)
        {
            std::cout << "Failed loading engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else
    {
        // ����������
        IBuilder *builder = createInferBuilder(gLogger);
        INetworkDefinition *network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IBuilderConfig *config = builder->createBuilderConfig();
        // config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 3UL << 32UL);
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 2UL << 30);
        if (FP_mode == "fp16")
        {
            printf("using kFP16!");
            config->setFlag(BuilderFlag::kFP16);
            // config->setFlag(BuilderFlag::kINT8);
        }
        else
        {
            printf("using kFP32!");
        }

        // ͨ��onnx�����������Ľ����������addConv�ķ�ʽ��䵽network��
        nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser->parseFromFile(onnxFile.c_str(), int(gLogger.reportable_severity)))
        {
            std::cout << std::string("Failed parsing .onnx file!") << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i)
            {
                auto *error = parser->getError(i);
                std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
            }
            return 1;
        }
        std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

        // 在构建配置中设置动态形状
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        profile->setDimensions("images", OptProfileSelector::kMIN, Dims4{1, 3, 640, 640});
        profile->setDimensions("images", OptProfileSelector::kOPT, Dims4{1, 3, 640, 640});
        profile->setDimensions("images", OptProfileSelector::kMAX, Dims4{1, 3, 640, 640});
        config->addOptimizationProfile(profile);

        // ����ָ���õ����ù������棬�õ����л�ģ��engineString
        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        if (engineString == nullptr || engineString->size() == 0)
        {
            std::cout << "Failed building serialized engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded building serialized engine!" << std::endl;

        // �����л���õ�engine
        IRuntime *runtime{createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr)
        {
            std::cout << "Failed building engine!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded building engine!" << std::endl;

        // ����engine
        std::ofstream engineFile(trtFile, std::ios::binary);
        if (!engineFile)
        {
            std::cout << "Failed opening file to write" << std::endl;
            return 1;
        }
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        if (engineFile.fail())
        {
            std::cout << "Failed saving .plan file!" << std::endl;
            return 1;
        }
        std::cout << "Succeeded saving .plan file!" << std::endl;
    }
    // ִ������
    // �����ִ��context
    IExecutionContext *context = engine->createExecutionContext();

    std::cout << std::string("Binding all? ") << std::string(context->allInputDimensionsSpecified() ? "Yes" : "No") << std::endl;
    int nBinding = engine->getNbBindings(); // ��ȡ���������������
    int nInput = 0;
    for (int i = 0; i < nBinding; ++i)
    {
        nInput += int(engine->bindingIsInput(i)); // �ۼ�������������
    }
    // int nOutput = nBinding - nInput;                    // �����������
    for (int i = 0; i < nBinding; ++i)
    {
        std::cout << std::string("Bind[") << i << std::string(i < nInput ? "]:i[" : "]:o[") << (i < nInput ? i : i - nInput) << std::string("]->");
        std::cout << dataTypeToString(engine->getBindingDataType(i)) << std::string(" ");
        std::cout << shapeToString(context->getBindingDimensions(i)) << std::string(" ");
        std::cout << engine->getBindingName(i) << std::endl;
    }

    // ��ӡ��������Ĵ�С
    std::vector<int> vBindingSize(nBinding, 0);
    for (int i = 0; i < nBinding; ++i)
    {
        Dims32 dim = context->getBindingDimensions(i);
        int size = 1;
        for (int j = 0; j < dim.nbDims; ++j)
        {
            size *= dim.d[j];
        }
        vBindingSize[i] = size * dataTypeToSize(engine->getBindingDataType(i));
        printf("id : %d, %d\n", i, vBindingSize[i]);
    }

    std::vector<void *> vBufferH{nBinding, nullptr};
    std::vector<void *> vBufferD{nBinding, nullptr};
    for (int i = 0; i < nBinding; ++i)
    {
        vBufferH[i] = (void *)new char[vBindingSize[i]];
        memset(vBufferH[i], 0, vBindingSize[i]); // FIXME
        CHECK_CUDA(cudaMalloc(&vBufferD[i], vBindingSize[i]));
    }

    for (int i = 0; i < nInput; ++i)
    {
        CHECK_CUDA(cudaMemcpy(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice));
    }

    context->executeV2(vBufferD.data()); // ͬ���������Ҫ�첽ִ�У�����context->enqueueV2(buffers, stream, nullptr);

    for (int i = nInput; i < nBinding; ++i)
    {
        CHECK_CUDA(cudaMemcpy(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i < nBinding; ++i)
    {
        CHECK_CUDA(cudaFree(vBufferD[i]));
    }

    return 0;
}