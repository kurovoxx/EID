import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def plot_potential_3d(f_u, x_range, y_range, xg, yg, obstacles, trajectory=None):
    """Genera una superficie 3D del potencial."""
    X, Y = np.meshgrid(x_range, y_range)
    
    # Manejo de errores numéricos durante la evaluación de la malla
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = f_u(X, Y)
        # Reemplazar posibles NaNs o Infs por valores muy altos pero finitos
        Z = np.nan_to_num(Z, nan=1000.0, posinf=1000.0, neginf=0.0)
    
    # Clipping para visualización estética
    z_max_vis = np.percentile(Z, 95)
    Z_vis = np.clip(Z, 0, z_max_vis) 
    
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z_vis, colorscale='Viridis', opacity=0.8)])
    
    # Marcador meta
    fig.add_trace(go.Scatter3d(x=[xg], y=[yg], z=[float(f_u(xg, yg))], 
                               mode='markers', marker=dict(size=10, color='red'), name='Meta'))
    
    # Marcadores obstáculos
    if obstacles:
        obs_x = [o[0] for o in obstacles]
        obs_y = [o[1] for o in obstacles]
        
        # Evaluamos cerca del obstáculo y limitamos su altura
        obs_z = []
        for o in obstacles:
            val = float(f_u(o[0] + 0.01, o[1] + 0.01))
            obs_z.append(min(val, z_max_vis))
            
        fig.add_trace(go.Scatter3d(x=obs_x, y=obs_y, z=obs_z, 
                                   mode='markers', marker=dict(size=8, color='black'), name='Obstáculos'))
        
    if trajectory is not None:
        # Evaluar trayectoria con seguridad
        with np.errstate(divide='ignore', invalid='ignore'):
            tz = f_u(trajectory[:, 0], trajectory[:, 1])
            tz = np.nan_to_num(tz, nan=z_max_vis, posinf=z_max_vis)
        
        fig.add_trace(go.Scatter3d(x=trajectory[:, 0], y=trajectory[:, 1], z=tz,
                                   mode='lines', line=dict(color='white', width=5), name='Trayectoria'))

    fig.update_layout(title='Superficie de Potencial Total U(x,y)',
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Potencial'),
                      margin=dict(l=0, r=0, b=0, t=40))
    return fig

def plot_contour_2d(f_u, f_grad_x, f_grad_y, x_range, y_range, xg, yg, obstacles, trajectory=None):
    """Genera un mapa de curvas de nivel y campo vectorial."""
    X, Y = np.meshgrid(x_range, y_range)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = f_u(X, Y)
        Z = np.nan_to_num(Z, nan=1000.0, posinf=1000.0)
    
    z_max_vis = np.percentile(Z, 95)
    Z_vis = np.clip(Z, 0, z_max_vis)
    
    fig = go.Figure()
    
    # Curvas de nivel
    fig.add_trace(go.Contour(x=x_range, y=y_range, z=Z_vis, colorscale='Viridis', 
                             contours_coloring='heatmap', name='Potencial'))
            
    # Meta y Obstáculos
    fig.add_trace(go.Scatter(x=[xg], y=[yg], mode='markers+text', 
                             marker=dict(size=15, color='red', symbol='star'), 
                             text="META", textposition="top center", name='Meta'))
    
    if obstacles:
        obs_x = [o[0] for o in obstacles]
        obs_y = [o[1] for o in obstacles]
        fig.add_trace(go.Scatter(x=obs_x, y=obs_y, mode='markers', 
                                 marker=dict(size=12, color='black', symbol='x'), name='Obstáculos'))
        
    if trajectory is not None:
        fig.add_trace(go.Scatter(x=trajectory[:, 0], y=trajectory[:, 1], 
                                 mode='lines+markers', line=dict(color='white', width=2), 
                                 marker=dict(size=4), name='Trayectoria'))

    fig.update_layout(title='Curvas de Nivel y Navegación',
                      xaxis_title='X', yaxis_title='Y',
                      width=700, height=600)
    return fig

def plot_iteration_analysis(f_u, x_range, y_range, xg, yg, obstacles, trajectory, grads, idx):
    """Muestra el estado específico de una iteración sobre el mapa."""
    X, Y = np.meshgrid(x_range, y_range)
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = f_u(X, Y)
        Z = np.nan_to_num(Z, nan=1000.0, posinf=1000.0)
    
    z_max_vis = np.percentile(Z, 95)
    Z_vis = np.clip(Z, 0, z_max_vis)
    
    fig = go.Figure()
    fig.add_trace(go.Contour(x=x_range, y=y_range, z=Z_vis, colorscale='Viridis', opacity=0.6))
    
    # Meta y Obstáculos
    fig.add_trace(go.Scatter(x=[xg], y=[yg], mode='markers', marker=dict(size=12, color='red', symbol='star'), name='Meta'))
    if obstacles:
        fig.add_trace(go.Scatter(x=[o[0] for o in obstacles], y=[o[1] for o in obstacles], 
                                 mode='markers', marker=dict(size=10, color='black', symbol='x'), name='Obstáculos'))

    # Trayectoria hasta el punto actual
    fig.add_trace(go.Scatter(x=trajectory[:idx+1, 0], y=trajectory[:idx+1, 1], 
                             mode='lines', line=dict(color='white', width=2), name='Trayectoria'))

    # Punto Actual
    cx, cy = trajectory[idx]
    fig.add_trace(go.Scatter(x=[cx], y=[cy], mode='markers', marker=dict(size=12, color='cyan', symbol='circle'), name='Robot'))

    # Vector Gradiente en el punto (si existe para esta iteración)
    if idx < len(grads):
        gx, gy = grads[idx]
        # Dibujar flecha (vector -gradiente, que es hacia donde se mueve)
        fig.add_annotation(x=cx - gx, y=cy - gy, ax=cx, ay=cy,
                           xref="x", yref="y", axref="x", ayref="y",
                           text="", showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=3, arrowcolor="red")
        
    fig.update_layout(title=f"Estado en Iteración {idx}", xaxis_title='X', yaxis_title='Y', width=700, height=600)
    return fig

def plot_gradient_norms(norms):
    """Grafica la norma del gradiente vs iteración."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=norms, mode='lines', line=dict(color='orange'), name='||∇U||'))
    fig.update_layout(title="Convergencia: Norma del Gradiente vs Iteración",
                      xaxis_title="Iteración", yaxis_title="||∇U||",
                      template="plotly_dark")
    return fig

