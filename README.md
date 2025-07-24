# Fundus Image Classification Workspace

This workspace provides a comprehensive environment for experimenting with different models and hyperparameters for fundus retina image classification. It's designed to make experimentation, monitoring, and evaluation as seamless as possible.

## Features

- **Model Experimentation**: Easily test different architectures and hyperparameters using a clean CLI interface
- **Comprehensive Monitoring**: Detailed WandB integration for tracking training metrics
- **Rich Terminal Feedback**: Real-time progress updates during training
- **Experiment Management**: Run multiple experiments with different configurations
- **Results Comparison**: Compare multiple runs to identify the best models
- **Thorough Evaluation**: Generate detailed reports and visualizations of model performance

## Project Structure

```
fundus-classification/
├── model/
│   ├── classifier/
│   │   ├── model.py       # Model architecture
│   │   └── dataset.py     # Data loading and processing
│   ├── losses/
│   │   └── losses.py      # Loss functions
│   └── utils/
│       └── metrics.py     # Evaluation metrics
├── experiments/
│   └── basic_sweep.json   # Sample experiment configuration
├── train.py               # Main training script
├── run_experiments.py     # Script for running multiple experiments
├── compare_results.py     # Script for comparing experiment results
└── evaluate_model.py      # Script for detailed model evaluation
```

## Getting Started

### 1. Training a Single Model

To train a single model with specific parameters:

```bash
python train.py --model_name efficientnet_b0 --batch_size 32 --img_size 224 --lr 0.001 --loss focal
```

This will train an EfficientNet B0 model with the specified hyperparameters, log results to WandB, and save checkpoints.

### 2. Running Multiple Experiments

To run a series of experiments with different configurations:

```bash
python run_experiments.py --config experiments/basic_sweep.json --experiment_name my_experiment
```

This will run all combinations of parameters specified in the JSON file and tag them with the experiment name for easy filtering.

### 3. Comparing Results

After running multiple experiments, compare the results to find the best configuration:

```bash
python compare_results.py --tag my_experiment --metric val_acc --group-by model_name
```

This will generate visualizations and tables comparing the performance of different models.

### 4. Evaluating a Model

For a detailed evaluation of your best model on a test set:

```bash
python evaluate_model.py --checkpoint checkpoints/best_model.ckpt --test_dir path/to/test_data --output_dir evaluation_results
```

This will generate a comprehensive report including confusion matrices, sample predictions, and detailed metrics.

## Advanced Usage

### Custom Loss Functions

You can choose between different loss functions:

- Cross-Entropy Loss: `--loss ce`
- Focal Loss: `--loss focal --focal_gamma 2.0`

### Learning Rate Schedulers

The model supports different learning rate schedulers:

```bash
python train.py --model_name resnet50 --lr 0.001 --scheduler cosine --scheduler_params '{"T_max": 10, "eta_min": 1e-6}'
```

### Data Augmentation

The data module includes standard augmentations. You can add more custom augmentations in `dataset.py`.

## Monitoring Training

### Terminal Output

The training script provides rich terminal output with:
- Training configuration summary
- Dataset information and class distribution
- Progress bars showing current epoch and estimated time
- Periodic metric updates
- Early stopping notifications
- Best model information

### WandB Integration

Each run automatically logs to WandB with:
- All hyperparameters
- Training and validation metrics (loss, accuracy, F1 score, etc.)
- Learning rate tracking
- Confusion matrices
- Sample predictions
- Model checkpoints

## Customization

### Adding New Models

The system uses `timm` library, so you can use any model supported by `timm` by specifying its name:

```bash
python train.py --model_name vit_base_patch16_224 --batch_size 16
```

### Adding New Metrics

To add a new metric, update the `model/utils/metrics.py` file and add your custom metrics.

## Tips for Effective Experimentation

1. **Start Small**: Begin with a few quick experiments to validate your pipeline
2. **Use Run Names**: Give meaningful names to your runs for easier identification
3. **Group Related Experiments**: Use the same tag for related experiments
4. **Track Multiple Metrics**: Don't rely solely on accuracy; monitor F1 score, precision, and recall
5. **Visualize Results**: Use the comparison tools to visualize trends across experiments
6. **Save Checkpoints**: Always save the best models for further evaluation and deployment

## Troubleshooting

### Common Issues

- **CUDA Out of Memory**: Reduce batch size or image size
- **Slow Training**: Check number of workers and batch size
- **Poor Performance**: Try different loss functions, learning rates, or model architectures
- **WandB Connection Issues**: Check your internet connection or run with `--offline` mode

## Requirements

- PyTorch
- PyTorch Lightning
- timm
- WandB
- scikit-learn
- pandas
- matplotlib
- seaborn
- rich (for terminal output)

Install requirements with:

```bash
pip install torch torchvision pytorch-lightning timm wandb scikit-learn pandas matplotlib seaborn rich
```

## License

This project is open-source and available for your personal and research use.