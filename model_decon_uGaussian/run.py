import torch
from model_decon import ModelDecon
from data_handler import GymDataHandler
from configs import config

def train_model(model, data_handler, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    for epoch in range(config['epochs']):
        states, actions, rewards, next_states, dones = data_handler.generate_batch(config['batch_size'])
        
        optimizer.zero_grad()
        loss, _, _, _ = model.neg_elbo(states, actions, rewards, dones, anneal=1)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    data_handler = GymDataHandler(config['env_name'])
    model = ModelDecon(config)
    train_model(model, data_handler, config)
