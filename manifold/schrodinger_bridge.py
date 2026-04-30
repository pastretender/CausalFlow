import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard time embedding to inject the temporal state 't' into the network.
    Crucial for continuous-time ODE/SDE modeling.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class CapitalFlowVectorField(nn.Module):
    """
    The Neural Vector Field v_theta(x, t).
    Predicts the 'velocity' and direction of capital rotation at state x and time t.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # Maps the concatenated state and time embedding to the vector field
        self.net = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t shape: (batch_size,)
        t_emb = self.time_mlp(t)
        # x shape: (batch_size, state_dim)
        xt = torch.cat([x, t_emb], dim=-1)
        return self.net(xt)

class SchrodingerFlowMatching(nn.Module):
    """
    Orchestrates the Flow Matching training and ODE integration inference
    to model the Schrödinger Bridge for capital rotation.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.vector_field = CapitalFlowVectorField(state_dim=state_dim)

    def compute_loss(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Computes the Flow Matching loss.
        x0: Current cross-sectional state (e.g., current concept activations)
        x1: Target future state (e.g., observed concept activations at t+horizon)
        """
        batch_size = x0.shape[0]

        # 1. Sample random time t ~ U[0, 1]
        t = torch.rand(batch_size, device=x0.device)

        # 2. Construct the optimal transport path x_t
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x0 + t_expand * x1

        # 3. The target vector field (velocity) is simply (x1 - x0)
        target_v = x1 - x0

        # 4. Predict velocity using the neural network
        pred_v = self.vector_field(x_t, t)

        # 5. MSE Loss between predicted and target velocity
        loss = torch.nn.functional.mse_loss(pred_v, target_v)
        return loss

    @torch.no_grad()
    def predict_trajectory(self, x0: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        Solves the ODE forward using the Euler method to predict the trajectory
        of capital rotation from current state x0.
        """
        device = x0.device
        batch_size = x0.shape[0]
        x_t = x0.clone()
        dt = 1.0 / steps

        trajectory = [x_t.clone()]

        for step in range(steps):
            t = torch.full((batch_size,), step * dt, device=device)
            v = self.vector_field(x_t, t)
            # Euler integration step: x_{t+dt} = x_t + v * dt
            x_t = x_t + v * dt
            trajectory.append(x_t.clone())

        return torch.stack(trajectory, dim=1) # Shape: (batch, steps+1, state_dim)

# --- Integration Example ---
# flow_model = SchrodingerFlowMatching(state_dim=50) # Assuming 50 CBM concepts
#
# # Training step
# loss = flow_model.compute_loss(current_concepts, future_concepts)
# loss.backward()
#
# # Live Trading Inference: Predict where capital is flowing in the next 10 steps
# predicted_path = flow_model.predict_trajectory(current_concepts, steps=10)
# final_predicted_state = predicted_path[:, -1, :]