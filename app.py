import streamlit as st
import numpy as np
import sympy as sp

import math_engine as me
import simulation as sim
import visualization as vis

st.set_page_config(page_title="Navegacion con Campos Potenciales", layout="wide")

st.title("Navegacion autonoma con campos potenciales")
st.write("""
Proyecto final de Calculo Avanzado. Acá simulamos un robot que se mueve en un plano 2D
usando un campo potencial: la meta lo atrae y los obstaculos lo repelen. El robot
se va moviendo siguiendo el -gradiente del potencial total (esto es lo que en el
enunciado llaman "direccion de maximo descenso").
""")

# --------------- parametros, todo esto va en la barra lateral ---------------
st.sidebar.header("Parametros")

st.sidebar.subheader("Posiciones")
x0 = st.sidebar.number_input("x inicial", value=0.0)
y0 = st.sidebar.number_input("y inicial", value=0.0)
xg = st.sidebar.number_input("x meta", value=10.0)
yg = st.sidebar.number_input("y meta", value=10.0)

st.sidebar.subheader("Constantes del modelo")
ka = st.sidebar.slider("ka (atraccion)", 0.1, 3.0, 1.0)
kr = st.sidebar.slider("kr (repulsion)", 1.0, 50.0, 15.0)

st.sidebar.subheader("Obstaculos")
n_obs = st.sidebar.number_input("Cantidad de obstaculos", min_value=0, max_value=6, value=2, step=1)

obstaculos = []
for i in range(int(n_obs)):
    col1, col2 = st.sidebar.columns(2)
    ox = col1.number_input("Obs " + str(i+1) + " - x", value=float(3 + i * 3), key="ox" + str(i))
    oy = col2.number_input("Obs " + str(i+1) + " - y", value=float(4 + i * 3), key="oy" + str(i))
    obstaculos.append((ox, oy))

st.sidebar.subheader("Simulacion")
alpha = st.sidebar.number_input("Paso (alpha)", value=0.1, step=0.01)
max_iter = st.sidebar.number_input("Iteraciones maximas", value=300, step=10)
tol = st.sidebar.number_input("Tolerancia para decir que llego a la meta", value=0.4, step=0.1)


# --------------- parte matematica (se recalcula cada vez que se cambia algo) ---------------
# esto de aca se ejecuta de nuevo cada vez que el usuario mueve un slider, ojo
# que sympy no es lo mas rapido del mundo pero para este tamaño de problema anda bien

Uatt, Urep, U = me.potencial_total(xg, yg, obstaculos, ka, kr)
dUdx, dUdy = me.calcular_gradiente(U)
fU, fdx, fdy = me.pasar_a_funcion_numerica(U, dUdx, dUdy)


tab_sim, tab_grad, tab_math, tab_calc, tab_teoria = st.tabs(["Simulacion", "Paso a paso", "Modelo matematico", "Analisis puntual", "Teoria"])


with tab_sim:
    st.subheader("Trayectoria del robot")

    trayectoria, gradientes, estado, mensaje = sim.simular_trayectoria(
        x0, y0, xg, yg, fdx, fdy, alpha, max_iter, tol
    )

    # dependiendo de como termino la simulacion mostramos un cartel distinto
    if estado == "exito":
        st.success(mensaje)
    elif estado == "minimo_local":
        st.warning(mensaje)
    elif estado == "colision":
        st.error(mensaje)
    else:
        st.info(mensaje)

    # el rango del grafico lo calculamos en base a donde estan los puntos
    # para que siempre se vea bien independiente de los valores que se pongan
    rango_max = max(xg, yg, x0, y0) + 4

    col1, col2 = st.columns(2)
    with col1:
        fig_contorno = vis.graficar_curvas_de_nivel(fU, xg, yg, obstaculos, trayectoria, rango=rango_max)
        st.plotly_chart(fig_contorno, use_container_width=True)
    with col2:
        fig_3d = vis.graficar_superficie_3d(fU, xg, yg, obstaculos, trayectoria, rango=rango_max)
        st.plotly_chart(fig_3d, use_container_width=True)

    if len(gradientes) > 0:
        st.subheader("Como fue convergiendo")
        fig_norma = vis.graficar_norma_gradiente(gradientes)
        st.plotly_chart(fig_norma, use_container_width=True)
        ult_gx, ult_gy = gradientes[-1]
        norma_final = np.sqrt(ult_gx**2 + ult_gy**2)
        st.write("Ultimo gradiente calculado: (" + str(round(ult_gx, 4)) + ", " + str(round(ult_gy, 4)) + "), norma = " + str(round(norma_final, 4)))


with tab_grad:
    st.subheader("Explorador paso a paso")
    st.write("""
    Con el slider de abajo uno puede ir viendo iteracion por iteracion como se mueve
    el robot, y la flecha naranja muestra para donde apunta el -gradiente en ese
    punto (que es justo hacia donde se va a mover en el siguiente paso).
    """)

    # recalculamos la trayectoria de nuevo aca (los mismos parametros de arriba)
    # se podria guardar de la pestaña anterior pero como streamlit reejecuta
    # todo el script de nuevo cada vez, es mas facil asi
    trayectoria2, gradientes2, estado2, mensaje2 = sim.simular_trayectoria(
        x0, y0, xg, yg, fdx, fdy, alpha, max_iter, tol
    )

    rango_max2 = max(xg, yg, x0, y0) + 4

    if len(trayectoria2) > 1:
        iter_idx = st.slider("Iteracion", 0, len(trayectoria2) - 1, len(trayectoria2) - 1)

        cx, cy = trayectoria2[iter_idx]
        u_val = float(fU(cx, cy))
        dist_meta = np.sqrt((cx - xg)**2 + (cy - yg)**2)

        if iter_idx < len(gradientes2):
            gx_i, gy_i = gradientes2[iter_idx]
            norma_i = np.sqrt(gx_i**2 + gy_i**2)
        else:
            gx_i, gy_i, norma_i = 0.0, 0.0, 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Potencial U aqui", str(round(u_val, 4)))
        m2.metric("Distancia a la meta", str(round(dist_meta, 4)))
        m3.metric("Norma del gradiente", str(round(norma_i, 4)))

        st.caption("Vector gradiente en este punto: (" + str(round(gx_i, 4)) + ", " + str(round(gy_i, 4)) + ")")

        fig_paso = vis.graficar_paso_a_paso(fU, xg, yg, obstaculos, trayectoria2, gradientes2, iter_idx, rango=rango_max2)
        st.plotly_chart(fig_paso, use_container_width=True)
    else:
        st.info("No hay suficientes pasos para mostrar (revisar los parametros de la simulacion).")


with tab_math:
    st.subheader("Funciones del modelo")
    st.write("Estas son las formulas que se usaron, son las mismas del enunciado del proyecto.")

    st.write("Potencial atractivo:")
    st.latex(sp.latex(Uatt))

    st.write("Potencial repulsivo (sumando todos los obstaculos):")
    st.latex(sp.latex(Urep))

    st.write("Potencial total U = U_atractivo + U_repulsivo:")
    st.latex(sp.latex(U))

    st.write("Gradiente de U (las derivadas parciales):")
    st.latex(r"\frac{\partial U}{\partial x} = " + sp.latex(dUdx))
    st.latex(r"\frac{\partial U}{\partial y} = " + sp.latex(dUdy))

    st.divider()
    st.write("**Puntos criticos**")
    st.write("""
    Aca tratamos de encontrar puntos donde el gradiente es (0,0), serian puntos
    de "equilibrio" del sistema. Como no es facil resolver esto a mano (mas
    encima con varios obstaculos la ecuacion queda fea), usamos el metodo de
    Newton, partiendo de un punto que uno elige.

    No siempre converge, ya que depende de que tan cerca este el punto inicial
    del punto critico real. Si no encuentra nada probar con otro punto cerca
    de la meta o cerca de algun obstaculo (ahi es donde mas se forman).
    """)

    col_a, col_b = st.columns(2)
    px_crit = col_a.number_input("Punto inicial x", value=float((x0 + xg) / 2))
    py_crit = col_b.number_input("Punto inicial y", value=float((y0 + yg) / 2))

    if st.button("Buscar punto critico"):
        resultado = me.buscar_punto_critico_cerca(dUdx, dUdy, px_crit, py_crit)
        if resultado is None:
            st.warning("No se encontro nada cerca de ese punto, el metodo no convergio. Probar con otro punto inicial.")
        else:
            st.write("Punto critico encontrado: (" + str(round(resultado['x'], 3)) + ", " + str(round(resultado['y'], 3)) + ")")
            st.write("Tipo: **" + resultado['tipo'] + "**")
            st.caption("(clasificado con el criterio de la segunda derivada, det(H) = " + str(round(resultado['det_hessiano'], 4)) + ")")


with tab_calc:
    st.subheader("Derivada direccional y plano tangente")
    st.write("Aca uno elige un punto cualquiera del mapa y una direccion, para ver como cambia el potencial justo ahi.")

    col1, col2 = st.columns(2)
    with col1:
        px = st.number_input("Punto x", value=float(x0), key="px2")
        py = st.number_input("Punto y", value=float(y0), key="py2")
    with col2:
        vx = st.number_input("Direccion vx", value=1.0)
        vy = st.number_input("Direccion vy", value=0.0)

    dd = me.derivada_direccional(dUdx, dUdy, px, py, vx, vy)
    if dd is None:
        st.error("El vector direccion no puede ser (0,0), no tiene sentido.")
    else:
        st.write("Derivada direccional en (" + str(px) + ", " + str(py) + ") en esa direccion: **" + str(round(dd, 4)) + "**")
        if dd > 0:
            st.caption("Sale positiva, el potencial aumenta yendo en esa direccion (nos alejamos de la zona buena).")
        else:
            st.caption("Sale negativa, el potencial disminuye en esa direccion (nos acercamos a la zona buena).")

    L, Up, gx_p, gy_p = me.plano_tangente(U, dUdx, dUdy, px, py)
    st.write("Aproximacion lineal (plano tangente) del potencial cerca de ese punto:")
    st.latex("L(x,y) = " + sp.latex(L))
    st.caption("U en ese punto: " + str(round(Up, 4)) + ", gradiente ahi: (" + str(round(gx_p, 4)) + ", " + str(round(gy_p, 4)) + ")")


with tab_teoria:
    st.subheader("Un resumen de la teoria detras de esto")
    st.write("""
    Esto es como un resumen rapido de los conceptos que usamos, para tenerlo
    a mano en la presentacion y no tener que explicarlo todo de memoria.
    """)

    st.markdown("""
    **1. Que es un campo potencial**

    La idea es tratar al robot como si fuera una particula que se mueve por
    un campo de fuerzas (esto se parece bastante a los campos electricos que
    se ven en fisica, por eso se llama "potencial"). La meta genera un "pozo"
    que atrae al robot, y los obstaculos generan "cerros" que lo empujan para
    el otro lado.

    **2. El gradiente**

    El gradiente de una funcion apunta hacia donde esa funcion crece mas rapido.
    Como nosotros queremos que el robot llegue al minimo (la meta), tiene que
    moverse para el lado contrario al gradiente, por eso en todos lados aparece
    el signo menos: -∇U.

    **3. Descenso de gradiente**

    Es el metodo que usamos para mover al robot paso a paso. La formula es:
    """)
    st.latex(r"(x,y)_{k+1} = (x,y)_k - \alpha \nabla U(x,y)_k")
    st.markdown("""
    donde alpha es el "tamaño de paso", si alpha es muy grande el robot puede
    saltarse la meta o moverse de forma rara, y si es muy chico la simulacion
    se demora muchas iteraciones en llegar (esto lo probamos cambiando el
    slider de alpha y se nota bastante la diferencia).

    **4. El problema de los minimos locales**

    A veces el robot puede quedar "atrapado" en un punto donde el gradiente es
    casi cero pero no es la meta, esto pasa generalmente cuando hay un obstaculo
    justo en la linea recta entre el inicio y la meta. Ahi las fuerzas de
    atraccion y repulsion se cancelan y el robot se queda pegado sin saber
    para donde ir. Es una de las limitaciones que tiene este metodo (lo
    mencionamos tambien en las conclusiones del informe).
    """)

    st.caption("Resumen hecho para el curso de Calculo Avanzado, no es 100% formal pero ayuda a explicar la idea.")