# Navegacion autonoma con campos potenciales

Proyecto final de Calculo Avanzado (MATE1189). La idea es simular un robot
que se mueve en un plano 2D usando un campo potencial: la meta lo atrae y
los obstaculos lo repelen, y se mueve siguiendo el -gradiente del potencial
total (osea, "bajando la pendiente" del potencial).

## Como correrlo

1. Instalar lo que se necesita:
```
pip install -r requirements.txt
```

2. Correr:
```
streamlit run app.py
```

3. Se deberia abrir solo en el navegador, si no en http://localhost:8501

## Que hace cada archivo

- app.py -> la interfaz, ahi se cambian los parametros y se ven los graficos
- math_engine.py -> toda la parte de calculo (potenciales, derivadas, puntos criticos)
- simulation.py -> el descenso de gradiente, mueve al robot paso a paso
- visualization.py -> los graficos (usamos plotly para que se puedan girar/hacer zoom)

## Pestañas de la app

- Simulacion: se ve la trayectoria completa del robot ya calculada
- Paso a paso: con un slider se puede ir viendo iteracion por iteracion, con
  la flecha del -gradiente mostrando para donde se mueve en cada paso
- Modelo matematico: las formulas y la busqueda de puntos criticos
- Analisis puntual: derivada direccional y plano tangente en un punto cualquiera

## Cosas que notamos al probar

- Si se pone un obstaculo justo en el medio entre el inicio y la meta, el robot
  puede quedar atrapado en un minimo local y nunca llegar (esto lo explicamos
  en el informe, es uno de los problemas conocidos de este metodo).
- La busqueda de puntos criticos (metodo de newton) depende mucho del punto
  inicial que se le da. Si no encuentra nada hay que probarlo mas cerca de
  donde uno cree que esta el punto critico (cerca de la meta o de algun
  obstaculo, que es donde mas aparecen).
- Si el robot pasa muy cerca de un obstaculo el programa detecta que el
  gradiente se va a infinito y para la simulacion ahi mismo (no la deja
  seguir con valores nan, eso nos pasaba al principio y se veian graficos rotos).
