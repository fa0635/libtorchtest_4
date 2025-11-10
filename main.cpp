#include <torch/torch.h>
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <random>
#include <cmath>

template <typename ActivationType = torch::nn::Tanh,
typename EndActivationType = torch::nn::Identity>
class MLPImpl final : public torch::nn::Module
{
public:
    MLPImpl(const int input_size,
            const std::vector<int>& hidden_sizes,
            const int output_size,
            const double dropout_prob = 0.0,
            const bool use_layer_norm = true)
    {
        if (input_size < 1)
        {
            throw std::invalid_argument("MLPImpl::MLPImpl: input_size cannot be less than one.");
        }

        if (output_size < 1)
        {
            throw std::invalid_argument("MLPImpl::MLPImpl: output_size cannot be less than one.");
        }

        if(hidden_sizes.empty())
        {
            throw std::invalid_argument("MLPImpl::MLPImpl: hidden_sizes cannot be empty.");
        }

        for(auto& it : hidden_sizes)
        {
            if (it < 1)
            {
                throw std::invalid_argument("MLPImpl::MLPImpl: All components of hidden_sizes must be greater than zero.");
            }
        }

        model = register_module("model", torch::nn::Sequential());

        model->push_back(torch::nn::Linear(input_size, hidden_sizes[0]));

        if (use_layer_norm)
        {
            model->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_sizes[0]})));
        }

        model->push_back(ActivationType());

        if (dropout_prob > 0.0)
        {
            model->push_back(torch::nn::Dropout(dropout_prob));
        }

        for (size_t i = 1; i < hidden_sizes.size(); ++i)
        {
            model->push_back(torch::nn::Linear(hidden_sizes[i-1], hidden_sizes[i]));

            if (use_layer_norm)
            {
                model->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_sizes[i]})));
            }

            model->push_back(ActivationType());

            if (dropout_prob > 0.0)
            {
                model->push_back(torch::nn::Dropout(dropout_prob));
            }
        }

        model->push_back(torch::nn::Linear(hidden_sizes.back(), output_size));

        if constexpr(!std::is_same_v<EndActivationType, torch::nn::Identity>)
        {
            model->push_back(EndActivationType());
        }
    }

    ~MLPImpl() override = default;

    torch::Tensor forward(torch::Tensor x)
    {
        return model->forward(x);
    }

private:
    torch::nn::Sequential model{nullptr};
};

template <typename ActivationType = torch::nn::Tanh, typename EndActivationType = torch::nn::Identity>
class MLP : public torch::nn::ModuleHolder<MLPImpl<ActivationType, EndActivationType>>
{
public:
    using torch::nn::ModuleHolder<MLPImpl<ActivationType, EndActivationType>>::ModuleHolder;
    using Impl TORCH_UNUSED_EXCEPT_CUDA = MLPImpl<ActivationType, EndActivationType>;
};

template <typename ActivationType = torch::nn::Tanh,
typename EndActivationType = torch::nn::Identity>
class GATConvImpl : public torch::nn::Module
{
public:
    GATConvImpl(const int input_node_attr_size,
                const std::vector<int>& hidden_sizes,
                const int output_node_attr_size,
                const int initial_node_attr_size,
                const int edge_attr_size,
                const double dropout_prob = 0.0,
                const bool use_layer_norm = true)
    {
        if (input_node_attr_size < 1)
        {
            throw std::invalid_argument("GATConvImpl::GATConvImpl: input_node_attr_size cannot be less than one.");
        }

        if (output_node_attr_size < 1)
        {
            throw std::invalid_argument("GATConvImpl::GATConvImpl: output_node_attr_size cannot be less than one.");
        }

        if (initial_node_attr_size < 1)
        {
            throw std::invalid_argument("GATConvImpl::GATConvImpl: initial_node_attr_size cannot be less than one.");
        }

        if (edge_attr_size < 1)
        {
            throw std::invalid_argument("GATConvImpl::GATConvImpl: edge_attr_size cannot be less than one.");
        }

        if(hidden_sizes.empty())
        {
            throw std::invalid_argument("GATConvImpl::GATConvImpl: hidden_sizes cannot be empty.");
        }

        for(auto& it : hidden_sizes)
        {
            if (it < 1)
            {
                throw std::invalid_argument("GATConvImpl::GATConvImpl: All components of hidden_sizes must be greater than zero.");
            }
        }

        mlp = register_module("mlp", MLP<ActivationType, EndActivationType>(5 * input_node_attr_size + initial_node_attr_size + 4 * edge_attr_size,
                                                                            hidden_sizes,
                                                                            output_node_attr_size,
                                                                            dropout_prob,
                                                                            use_layer_norm));
    }

    virtual ~GATConvImpl() override = default;

    virtual torch::Tensor forward(torch::Tensor edge_index, torch::Tensor node_attr,
                                  torch::Tensor edge_attr, torch::Tensor edge_weight, torch::Tensor initial_node_attr)
    {
        auto reversed_edge_index = edge_index.flip(0);

        auto one_hop_incoming = propagate(edge_index, node_attr, edge_attr, edge_weight, 1);
        auto one_hop_outgoing = propagate(reversed_edge_index, node_attr, edge_attr, edge_weight, 1);

        auto two_hop_incoming = propagate(edge_index, one_hop_incoming, edge_attr, edge_weight, 2);
        auto two_hop_outgoing = propagate(reversed_edge_index, one_hop_outgoing, edge_attr, edge_weight, 2);

        auto combined = torch::cat({initial_node_attr, node_attr,
            one_hop_incoming, one_hop_outgoing,
            two_hop_incoming, two_hop_outgoing}, -1);

        return mlp->forward(combined);
    }

protected:
    virtual torch::Tensor propagate(torch::Tensor edge_index, torch::Tensor node_attr,
                                    torch::Tensor edge_attr, torch::Tensor edge_weight, int hop)
    {
        auto messages = message(edge_index, node_attr, edge_attr, edge_weight, hop);

        return aggregate(edge_index, messages, node_attr.size(0));
    }

    virtual torch::Tensor message(torch::Tensor edge_index, torch::Tensor node_attr,
                                  torch::Tensor edge_attr, torch::Tensor edge_weight, int hop)
    {
        auto source_nodes = edge_index[0];
        auto node_attr_j = node_attr.index_select(0, source_nodes);

        if(hop == 1)
        {
            return edge_weight * torch::cat({node_attr_j, edge_attr}, -1);
        }
        else
        {
            return edge_weight * node_attr_j;
        }
    }

    virtual torch::Tensor aggregate(torch::Tensor edge_index, torch::Tensor messages, int num_nodes)
    {
        auto target_nodes = edge_index[1];

        return torch::zeros({num_nodes, messages.size(1)}, messages.options()).index_add_(0, target_nodes, messages);
    }

    MLP<ActivationType, EndActivationType> mlp{nullptr};
};

template <typename ActivationType = torch::nn::Tanh, typename EndActivationType = torch::nn::Identity>
class GATConv : public torch::nn::ModuleHolder<GATConvImpl<ActivationType, EndActivationType>>
{
public:
    using torch::nn::ModuleHolder<GATConvImpl<ActivationType, EndActivationType>>::ModuleHolder;
    using Impl TORCH_UNUSED_EXCEPT_CUDA = GATConvImpl<ActivationType, EndActivationType>;
};

template <typename ActivationType = torch::nn::Tanh,
typename EndActivationType = torch::nn::Identity>
class NNImpl : public torch::nn::Module
{
public:
    NNImpl(const int node_attr_size,
           const std::vector<int>& hidden_sizes_1,
           const std::vector<int>& hidden_sizes_2,
           const std::vector<int>& hidden_sizes_mlp,
           const int output_node_attr_size,
           const int edge_attr_size,
           const double dropout_prob = 0.0,
           const bool use_layer_norm = true,
           const int k = 6)
    {
        gatconv1 = register_module("gatconv1", GATConv<ActivationType, EndActivationType>(node_attr_size,
                                                                                          hidden_sizes_1,
                                                                                          output_node_attr_size,
                                                                                          node_attr_size,
                                                                                          edge_attr_size,
                                                                                          dropout_prob,
                                                                                          use_layer_norm));
        gatconv2 = register_module("gatconv2", GATConv<ActivationType, EndActivationType>(output_node_attr_size,
                                                                                          hidden_sizes_2,
                                                                                          output_node_attr_size,
                                                                                          node_attr_size,
                                                                                          edge_attr_size,
                                                                                          dropout_prob,
                                                                                          use_layer_norm));
        mlp = register_module("mlp", MLP<ActivationType, EndActivationType>(2 * output_node_attr_size,
                                                                            hidden_sizes_mlp,
                                                                            1,
                                                                            dropout_prob,
                                                                            use_layer_norm));
        this->k = k;
    }
    virtual ~NNImpl() override = default;
    virtual torch::Tensor forward(torch::Tensor edge_index, torch::Tensor node_attr,
                                  torch::Tensor edge_attr, torch::Tensor edge_weight)
    {
        torch::Tensor output_node_attr = gatconv1->forward(edge_index, node_attr, edge_attr, edge_weight, node_attr);
        for (int i = 0; i < k - 1; ++i)
        {
            output_node_attr = gatconv2->forward(edge_index, output_node_attr, edge_attr, edge_weight, node_attr);
        }
        auto source_nodes = edge_index[0];
        auto node_attr_1 = output_node_attr.index_select(0, source_nodes);
        auto target_nodes = edge_index[1];
        auto node_attr_2 = output_node_attr.index_select(0, target_nodes);
        auto output_edge_attr = torch::cat({node_attr_1, node_attr_2}, -1);
        return mlp->forward(output_edge_attr);
    }
protected:
    GATConv<ActivationType, EndActivationType> gatconv1{nullptr};
    GATConv<ActivationType, EndActivationType> gatconv2{nullptr};
    MLP<ActivationType, EndActivationType> mlp{nullptr};
    int k;
};

template <typename ActivationType = torch::nn::Tanh, typename EndActivationType = torch::nn::Identity>
class NN : public torch::nn::ModuleHolder<NNImpl<ActivationType, EndActivationType>>
{
public:
    using torch::nn::ModuleHolder<NNImpl<ActivationType, EndActivationType>>::ModuleHolder;
    using Impl TORCH_UNUSED_EXCEPT_CUDA = NNImpl<ActivationType, EndActivationType>;
};

int main()
{
    auto start = std::chrono::steady_clock::now();
    
    YAML::Node config = YAML::LoadFile("../configs/training_parameters.yaml");
    int num_epochs = config["num_epochs"].as<int>();
    float lr = config["lr"].as<float>();

    std::mt19937 gen;
    std::normal_distribution d{30.0, 3.0};
    std::vector<torch::Tensor> graph_node_features(100);
    std::vector<torch::Tensor> graph_edge_index(100);
    std::vector<torch::Tensor> graph_edge_features(100);
    std::vector<torch::Tensor> graph_edge_labels(100);
    std::vector<torch::Tensor> graph_edge_weights(100);
    for (int i = 0; i < 100; ++i)
    {
        int graph_size = std::lround(d(gen));
        graph_node_features[i] = torch::rand({graph_size, 3});
        torch::Tensor adjacency_matrix = torch::rand({graph_size, graph_size});
        graph_edge_index[i] = torch::argwhere(adjacency_matrix.tril(-1) > 0.7).transpose(0, 1);
        graph_edge_features[i] = graph_node_features[i].index_select(0, graph_edge_index[i][0]) - graph_node_features[i].index_select(0, graph_edge_index[i][1]);
        graph_edge_labels[i] = torch::rand({graph_edge_index[i].size(1), 1});
        graph_edge_weights[i] = torch::ones({graph_edge_index[i].size(1), 1});
    }

    std::vector<int> hidden_sizes = {64, 64};
    std::vector<int> hidden_sizes_mlp = {80, 80};
    auto model = NN<torch::nn::ReLU, torch::nn::Identity>(3, hidden_sizes, hidden_sizes, hidden_sizes_mlp, 32, 3);
    torch::optim::Adam opt(model->parameters(), lr);
    torch::nn::MSELoss loss_fn;
    torch::nn::L1Loss metric_fn;
    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        float epoch_loss = 0;
        float epoch_metric = 0;
        for (int i = 0; i < 100; ++i)
        {
            torch::Tensor pred = model->forward(graph_edge_index[i], graph_node_features[i], graph_edge_features[i], graph_edge_weights[i]);
            torch::Tensor loss = loss_fn(pred, graph_edge_labels[i]);
            torch::Tensor metric = metric_fn(pred, graph_edge_labels[i]);
            loss.backward();
            opt.step();
            opt.zero_grad();
            epoch_loss += loss.item<float>();
            epoch_metric += metric.item<float>();
        }
        std::cout << "epoch:\t" << epoch << ";\tloss:\t" << epoch_loss / 100 << ";\tmetric:\t" << epoch_metric / 100 << '\n';
    }
    
    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Total CPU/GPU time: " << elapsed.count() << " s.\n";
    
    return 0;
}
