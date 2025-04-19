import os
import argparse
import yaml
import wandb
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()


def create_sweep(args):
    """Create a sweep with the specified configuration."""
    # Load sweep configuration
    with open(args.config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Update sweep configuration with command-line arguments
    if args.method:
        sweep_config['method'] = args.method
    
    if args.metric_name:
        if 'metric' not in sweep_config:
            sweep_config['metric'] = {}
        sweep_config['metric']['name'] = args.metric_name
    
    if args.metric_goal:
        if 'metric' not in sweep_config:
            sweep_config['metric'] = {}
        sweep_config['metric']['goal'] = args.metric_goal
    
    # Update program path
    sweep_config['program'] = 'rating/train.py'
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    
    print(f"Sweep created with ID: {sweep_id}")
    print(f"To start an agent, run: wandb agent {args.project}/{sweep_id}")
    
    # Start agent if requested
    if args.start_agent:
        print(f"Starting {args.num_agents} agent(s)...")
        for _ in range(args.num_agents):
            wandb.agent(sweep_id, project=args.project)
    
    return sweep_id


def main():
    parser = argparse.ArgumentParser(description='Create a hyperparameter sweep')
    
    # Sweep parameters
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to sweep configuration file')
    parser.add_argument('--project', type=str, default='recommendation_system',
                        help='Weights & Biases project name')
    parser.add_argument('--method', type=str, choices=['grid', 'random', 'bayes'],
                        help='Search method (grid, random, or bayes)')
    parser.add_argument('--metric_name', type=str,
                        help='Metric to optimize')
    parser.add_argument('--metric_goal', type=str, choices=['minimize', 'maximize'],
                        help='Goal for the metric (minimize or maximize)')
    
    # Agent parameters
    parser.add_argument('--start_agent', action='store_true',
                        help='Start agent(s) after creating sweep')
    parser.add_argument('--num_agents', type=int, default=1,
                        help='Number of agents to start')
    
    args = parser.parse_args()
    
    create_sweep(args)


if __name__ == "__main__":
    main()