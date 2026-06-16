import streamlit as st
import numpy as np
import sympy as sp
import math_engine as me
import simulation as sim
import visualization as vis

st.set_page_config(page_title="Simulador de Navegación - Campos Potenciales", layout="wide")

st.title("🚀 Simulación de Navegación Autónoma con Campos Potenciales")
st.markdown("""
Esta aplicación simula el movimiento de un robot en un entorno 2D utilizando **Campos Potenciales Artificiales**. 
Se aplican conceptos de **Cálculo Multivariable**: gradientes, derivadas parciales y descenso de gradiente.
""")

# --- SIDEBAR: Parámetros ---
st.sidebar.header("⚙️ Configuración del Entorno")

# Posiciones
x0 = st.sidebar.number_input("Posición inicial X (x0)", value=6.0)
y0 = st.sidebar.number_input("Posición inicial Y (y0)", value=0.0)
xg = st.sidebar.number_input("Meta X (xg)", value=10.0)
yg = st.sidebar.number_input("Meta Y (yg)", value=10.0)

# Parámetros Modelo
ka = st.sidebar.slider("Constante Atracción (ka)", 0.1, 5.0, 1.0)
kr = st.sidebar.slider("Constante Repulsión (kr)", 1.0, 100.0, 20.0)

# Obstáculos
n_obs = st.sidebar.number_input("Número de Obstáculos", min_value=0, max_value=10, value=2)
obstacles = []
for i in range(n_obs):
    st.sidebar.subheader(f"Obstáculo {i+1}")
    ox = st.sidebar.number_input(f"X_{i+1}", value=float(3 + i*2), key=f"ox_{i}")
    oy = st.sidebar.number_input(f"Y_{i+1}", value=float(3 + i*2), key=f"oy_{i}")
    obstacles.append((ox, oy, 1.0))

# Parámetros Simulación
st.sidebar.header("🏃 Parámetros de Simulación")
alpha = st.sidebar.number_input("Tamaño de paso (alpha)", value=0.1, step=0.01)
max_iter = st.sidebar.number_input("Máx. Iteraciones", value=200)
tol = st.sidebar.number_input("Tolerancia Meta", value=0.5)

# --- CÁLCULO SIMBÓLICO ---
x_sym, y_sym, u_att, u_rep, u_total = me.create_symbolic_potentials(xg, yg, obstacles, ka, kr)
grad_x, grad_y = me.calculate_gradients(u_total, x_sym, y_sym)
f_u, f_grad_x, f_grad_y = me.get_numerical_functions(u_total, grad_x, grad_y, x_sym, y_sym)

# --- TABS ---
tab1, tab_grad, tab2, tab3, tab4 = st.tabs(["📊 Simulación", "📈 Análisis Gradiente", "🧮 Matemáticas", "🔬 Cálculo", "📝 Teoría"])

with tab1:
    st.header("Visualización de la Navegación")
    
    # Ejecutar simulación
    history, result, grads, norms = sim.run_simulation(x0, y0, xg, yg, f_grad_x, f_grad_y, alpha, max_iter, tol, obstacles)
    
    # Mostrar mensaje de estado elegante
    if result["type"] == "success":
        st.success(result["message"])
    elif result["type"] == "local_minimum":
        st.warning(result["message"])
    elif result["type"] == "max_iter":
        st.info(result["message"])
    else:
        st.error(result["message"])
        
    # Mostrar Gradiente Final
    g_x, g_y = result["final_grad"]
    g_norm = result["grad_norm"]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Final ∂U/∂x", f"{g_x:.4f}")
    m2.metric("Final ∂U/∂y", f"{g_y:.4f}")
    m3.metric("Norma ||∇U||", f"{g_norm:.4f}")
    
    col1, col2 = st.columns(2)
    
    # Rangos para la malla
    x_range = np.linspace(min(x0, xg, -2), max(x0, xg, 12), 50)
    y_range = np.linspace(min(y0, yg, -2), max(y0, yg, 12), 50)
    
    with col1:
        fig_contour = vis.plot_contour_2d(f_u, f_grad_x, f_grad_y, x_range, y_range, xg, yg, obstacles, trajectory=history)
        st.plotly_chart(fig_contour, use_container_width=True)
        
    with col2:
        fig_3d = vis.plot_potential_3d(f_u, x_range, y_range, xg, yg, obstacles, trajectory=history)
        st.plotly_chart(fig_3d, use_container_width=True)

    # --- EXPLORADOR INTERACTIVO ---
    st.divider()
    st.subheader("🕵️ Explorador Paso a Paso")
    iter_idx = st.slider("Seleccionar Iteración", 0, len(history)-1, len(history)-1)
    
    curr_pos = history[iter_idx]
    
    # Calcular valores para esta iteración
    u_val = float(f_u(curr_pos[0], curr_pos[1]))
    dist_goal = np.sqrt((curr_pos[0]-xg)**2 + (curr_pos[1]-yg)**2)
    
    if iter_idx < len(grads):
        gx_val, gy_val = grads[iter_idx]
        gn_val = norms[iter_idx]
    else:
        gx_val, gy_val, gn_val = 0, 0, 0

    col_metrics = st.columns(4)
    col_metrics[0].metric("Potencial U", f"{u_val:.4f}")
    col_metrics[1].metric("|U| (Valor Abs)", f"{abs(u_val):.4f}")
    col_metrics[2].metric("Distancia Meta", f"{dist_goal:.4f}")
    col_metrics[3].metric("Norma Gradiente", f"{gn_val:.4f}")
    
    st.markdown(f"**Vector Gradiente en este punto:** $\\nabla U = ({gx_val:.4f}, {gy_val:.4f})$")
    
    fig_iter = vis.plot_iteration_analysis(f_u, x_range, y_range, xg, yg, obstacles, history, grads, iter_idx)
    st.plotly_chart(fig_iter, use_container_width=True)

with tab_grad:
    st.header("Análisis de Convergencia")
    if len(norms) > 0:
        fig_norms = vis.plot_gradient_norms(norms)
        st.plotly_chart(fig_norms, use_container_width=True)
    else:
        st.info("No hay datos de gradiente para mostrar.")


with tab2:
    st.header("Expresiones Matemáticas (SymPy)")
    st.markdown("A continuación se presentan las funciones de potencial y sus gradientes calculados analíticamente.")
    
    st.subheader("Potencial Total $U(x,y)$")
    st.latex(sp.latex(u_total))
    
    st.subheader("Gradiente $\\nabla U$")
    st.markdown("Componente $\\frac{\\partial U}{\\partial x}$:")
    st.latex(sp.latex(grad_x))
    st.markdown("Componente $\\frac{\\partial U}{\\partial y}$:")
    st.latex(sp.latex(grad_y))
    
    # Clasificación de puntos críticos (opcional/simplificado)
    if st.button("Buscar puntos críticos"):
        st.warning("La resolución analítica de $\\nabla U = 0$ puede ser compleja para múltiples obstáculos.")
        # Aquí se podría añadir un solver numérico para puntos críticos

with tab3:
    st.header("Análisis de Cálculo Multivariable")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Derivada Direccional")
        px = st.number_input("Punto de análisis X", value=float(x0))
        py = st.number_input("Punto de análisis Y", value=float(y0))
        dx = st.number_input("Dirección vector v_x", value=1.0)
        dy = st.number_input("Dirección vector v_y", value=0.0)
        
        # Normalizar vector
        norm = np.sqrt(dx**2 + dy**2)
        if norm > 0:
            ux, uy = dx/norm, dy/norm
            dd = me.directional_derivative(u_total, x_sym, y_sym, px, py, ux, uy)
            st.success(f"La derivada direccional en ({px}, {py}) en dirección ({ux:.2f}, {uy:.2f}) es: **{dd:.4f}**")
        else:
            st.error("El vector de dirección no puede ser nulo.")

    with col_b:
        st.subheader("Linealización y Plano Tangente")
        st.markdown(f"Aproximación lineal en el punto seleccionando en 'Derivada Direccional': ({px}, {py})")
        L_expr = me.linearization(u_total, x_sym, y_sym, px, py)
        st.latex(f"L(x,y) = {sp.latex(L_expr)}")
        
        st.info("El plano tangente representa la mejor aproximación lineal a la superficie en ese punto.")

with tab4:
    st.header("Fundamentos Teóricos")
    st.markdown("""
    ### 1. Campos Potenciales
    El robot es tratado como una partícula bajo la influencia de un campo de fuerzas.
    - **Meta:** Crea un pozo de potencial (atracción).
    - **Obstáculos:** Crean picos de potencial (repulsión).
    
    ### 2. El Gradiente ($\nabla U$)
    El gradiente apunta en la dirección de **máximo crecimiento** de la función. Para llegar al objetivo (mínimo potencial), el robot se mueve en dirección opuesta: $-\nabla U$.
    
    ### 3. Descenso de Gradiente
    Es el algoritmo iterativo:
    $$(x, y)_{k+1} = (x, y)_k - \alpha \nabla U(x, y)_k$$
    donde $\alpha$ es el tamaño de paso.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Desarrollado para el curso de Cálculo Multivariable.")
