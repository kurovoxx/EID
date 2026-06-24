import numpy as np
import plotly.graph_objects as go

# graficos del proyecto. usamos plotly porque queriamos que se pudiera
# girar el grafico 3d con el mouse y hacer zoom, con matplotlib eso no se puede
# (matplotlib lo probamos primero pero quedaba todo plano y fijo, no se podia
# mover nada, asi que al final nos quedamos con plotly aunque cueste un poco mas)


def graficar_curvas_de_nivel(fU, xg, yg, obstaculos, trayectoria=None, rango=15):
    # esto dibuja el mapa de calor del potencial visto desde arriba, con las
    # curvas de nivel U(x,y)=c que pide el enunciado en la parte 6.2

    x_vals = np.linspace(-2, rango, 150)
    y_vals = np.linspace(-2, rango, 150)
    X, Y = np.meshgrid(x_vals, y_vals)

    # esto de errstate es pa que no tire warnings feos en la consola cuando
    # se evalua justo encima de un obstaculo (da inf o nan ahi)
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = fU(X, Y)

    # recortamos los valores muy grandes, si no el grafico sale todo de
    # un solo color porque cerca del obstaculo el potencial es gigante
    Z = np.nan_to_num(Z, nan=50, posinf=50)
    Z_vis = np.clip(Z, 0, 50)

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=x_vals, y=y_vals, z=Z_vis,
        colorscale='Viridis', contours_coloring='heatmap', name='U(x,y)'
    ))

    fig.add_trace(go.Scatter(
        x=[xg], y=[yg], mode='markers+text', text=['Meta'], textposition='top center',
        marker=dict(size=14, color='red', symbol='star'), name='Meta'
    ))

    if obstaculos:
        obs_x = [o[0] for o in obstaculos]
        obs_y = [o[1] for o in obstaculos]
        fig.add_trace(go.Scatter(
            x=obs_x, y=obs_y, mode='markers',
            marker=dict(size=12, color='black', symbol='x'), name='Obstaculos'
        ))

    if trayectoria is not None:
        fig.add_trace(go.Scatter(
            x=trayectoria[:, 0], y=trayectoria[:, 1], mode='lines+markers',
            line=dict(color='white', width=2), marker=dict(size=4), name='Trayectoria'
        ))

    fig.update_layout(
        title='Curvas de nivel del potencial U(x,y)',
        xaxis_title='x', yaxis_title='y',
        width=650, height=600
    )

    return fig


def graficar_superficie_3d(fU, xg, yg, obstaculos, trayectoria=None, rango=15):
    # lo mismo pero en 3d, esta parte la pide el enunciado en 6.9
    # (graficar la superficie e identificar obstaculos y meta)

    x_vals = np.linspace(-2, rango, 60)
    y_vals = np.linspace(-2, rango, 60)
    X, Y = np.meshgrid(x_vals, y_vals)

    with np.errstate(divide='ignore', invalid='ignore'):
        Z = fU(X, Y)
    Z = np.nan_to_num(Z, nan=50, posinf=50)
    Z_vis = np.clip(Z, 0, 50)

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z_vis, colorscale='Viridis', opacity=0.85)])

    # marcamos la meta en la superficie (con z=0 para que se vea abajo,
    # si la pegamos a la superficie misma queda media escondida)
    fig.add_trace(go.Scatter3d(
        x=[xg], y=[yg], z=[0], mode='markers',
        marker=dict(size=8, color='red'), name='Meta'
    ))

    if obstaculos:
        obs_x = [o[0] for o in obstaculos]
        obs_y = [o[1] for o in obstaculos]
        obs_z = [0 for _ in obstaculos]
        fig.add_trace(go.Scatter3d(
            x=obs_x, y=obs_y, z=obs_z, mode='markers',
            marker=dict(size=6, color='black'), name='Obstaculos'
        ))

    if trayectoria is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            z_tray = fU(trayectoria[:, 0], trayectoria[:, 1])
        z_tray = np.nan_to_num(z_tray, nan=50, posinf=50)
        z_tray = np.clip(z_tray, 0, 50)
        fig.add_trace(go.Scatter3d(
            x=trayectoria[:, 0], y=trayectoria[:, 1], z=z_tray, mode='lines',
            line=dict(color='white', width=5), name='Trayectoria'
        ))

    fig.update_layout(
        title='Superficie del potencial total',
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='U(x,y)'),
        width=650, height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig


def graficar_norma_gradiente(gradientes):
    # grafico extra para ver si la simulacion va convergiendo o se quedo
    # estancada, la norma deberia ir bajando si todo anda bien
    normas = np.sqrt(gradientes[:, 0]**2 + gradientes[:, 1]**2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=normas, mode='lines', line=dict(color='darkorange'), name='||grad U||'))
    fig.update_layout(
        title='Norma del gradiente vs iteracion',
        xaxis_title='Iteracion', yaxis_title='||grad U||',
        width=650, height=350
    )

    return fig


def graficar_paso_a_paso(fU, xg, yg, obstaculos, trayectoria, gradientes, idx, rango=15):
    # esto es para el explorador paso a paso, muestra el mapa de curvas de nivel
    # pero solo hasta la iteracion "idx" que se elija con el slider, y dibuja
    # una flecha con el vector -gradiente (que es hacia donde se mueve el robot)

    x_vals = np.linspace(-2, rango, 150)
    y_vals = np.linspace(-2, rango, 150)
    X, Y = np.meshgrid(x_vals, y_vals)

    with np.errstate(divide='ignore', invalid='ignore'):
        Z = fU(X, Y)
    Z = np.nan_to_num(Z, nan=50, posinf=50)
    Z_vis = np.clip(Z, 0, 50)

    fig = go.Figure()

    fig.add_trace(go.Contour(x=x_vals, y=y_vals, z=Z_vis, colorscale='Viridis', opacity=0.6, showscale=False))

    fig.add_trace(go.Scatter(
        x=[xg], y=[yg], mode='markers', marker=dict(size=14, color='red', symbol='star'), name='Meta'
    ))

    if obstaculos:
        obs_x = [o[0] for o in obstaculos]
        obs_y = [o[1] for o in obstaculos]
        fig.add_trace(go.Scatter(
            x=obs_x, y=obs_y, mode='markers', marker=dict(size=12, color='black', symbol='x'), name='Obstaculos'
        ))

    # la trayectoria recorrida hasta este punto nomas (no toda la trayectoria completa)
    fig.add_trace(go.Scatter(
        x=trayectoria[:idx+1, 0], y=trayectoria[:idx+1, 1], mode='lines',
        line=dict(color='white', width=2), name='Recorrido'
    ))

    cx, cy = trayectoria[idx]
    fig.add_trace(go.Scatter(
        x=[cx], y=[cy], mode='markers', marker=dict(size=13, color='cyan'), name='Robot ahora'
    ))

    # la flecha del -gradiente, que es para donde se va a mover en el siguiente paso
    if idx < len(gradientes):
        gx, gy = gradientes[idx]
        fig.add_annotation(
            x=cx - gx, y=cy - gy, ax=cx, ay=cy,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=3, arrowcolor='orange'
        )

    fig.update_layout(
        title='Iteracion numero ' + str(idx),
        xaxis_title='x', yaxis_title='y',
        width=650, height=600
    )

    return fig
