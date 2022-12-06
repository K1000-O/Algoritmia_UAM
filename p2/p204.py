"""
    Implementación de las funciones pedidadas para la solución de la práctica 2 de la asignatura de
    Practicas de Algoritmos y Estructuras de Datos Avanzadas.

    Autor: Alejandro Raúl Hurtado <alejandror.hurtado@estudiante.uam.es>

    Autor: Camilo Jené Conde <camilo.jenec@estudiante.uam.es>

    2022 EPS-UAM 
"""
import numpy as np
import itertools

from typing import Dict, List

def init_cd(n: int)-> np.ndarray:
    """
    Nombre:
        init_cd
    
    Descripción:
        Inicializador de un array. Creamos un array de n elementos iniciados a -1.
    
    Argumentos:
        - n: tamaño del array.

    Retorno:
        - Array inicializado a -1.
    """
    return np.array([-1]*n)

def union(rep_1: int, rep_2: int, p_cd: np.ndarray)-> int:
    """
    Nombre:
        union
    
    Descripción:
        Función que une dos árboles. Debemos comprobar cuál es más profundo para intentar
        no aumentar la profundidad del árbol si no es necesario.
    
    Argumentos:
        - rep_1: índice de la raíz del primer árbol.
        - rep_2: índice de la raíz del segundo árbol.
        - p_cd: array con los árboles.

    Retorno:
        - La raíz de ambos árboles unidos.
    """
    if p_cd[rep_1] < p_cd[rep_2]:
        p_cd[rep_2] = rep_1
        return rep_1
    elif p_cd[rep_1] > p_cd[rep_2]:
        p_cd[rep_1] = rep_2
        return rep_2
    else:
        p_cd[rep_1] -= 1
        p_cd[rep_2] = rep_1
        return rep_1

def find(ind: int, p_cd: np.ndarray)-> int:
    """
    Nombre:
        find

    Descripción:
        Función encargada de encontrar la raíz del índice pasado por parámetro. Se
        pide la compresión de caminos para que el rendimiento al buscar otro elemento
        sea mayor.

    Argumentos:
        - ind: índice del nodo con el que vamos a trabajar.
        - p_cd: array con los árboles.
    
    Retorno:
        - Índice de la raíz del árbol perteneciente al nodo.
    """
    aux = []
    while p_cd[ind] >= 0:
        """
        Algoritmo de búsqueda hacía arriba.

        Guardamos en un array los nodos visitados para más tarde comprimir el camino.
        """
        aux.append(ind) # Guardamos el índice del nodo visitado.
        ind = p_cd[ind]

    for x in range(len(aux)):
        """
        Compresión de caminos. Situamos como padre de los nodos visitados a la 
        raíz obtenida.
        """
        p_cd[aux[x]] = ind

    return ind

def cd_2_dict(p_cd: np.ndarray)-> Dict:
    """
    Nombre:
        cd_2_dict
    
    Descripción:
        Función encargada de devolver un diccionaro compuesto por:
            (clave = raíz del árbol: valor = árbol)
    
    Argumentos:
        - p_cd: array con los árboles.
    
    Retorno:
        - Diccionario con los datos de los hijos de cada raíz.
    """
    dict = {}
    for x in range(len(p_cd)):
        if p_cd[x] < 0:
            if x not in dict:
                dict[x] = [x]
        else:
            if find(x, p_cd) in dict:
                dict[find(x, p_cd)] += [x]
            else:
                dict[find(x, p_cd)] = [find(x, p_cd)]
                dict[find(x, p_cd)] += [x]
    
    return dict

def ccs(n: int, l: List)-> Dict:
    """
    Nombre:
        css

    Descripción:
        Función creada para trabajar con componentes conexas.
        Sobre la lista, crearemos un diccionario con la siguiente estructura:

            - clave: el padre.
            - valor: la conexión.
            - {valor1: (valor1, valor2)}

    Argumentos:
        - n: el número de nodos que hay.
        - l: la lista con las conexiones.
    Retorno:
        - Diccionario con las componentes conexas.
    """
    d = {}
    list_aux = init_cd(n)

    for k, v in l:
        rK = find(k, list_aux)
        rV = find(v, list_aux)

        if rK != rV:
            if k in d:
                d[k] += [(k, v)]
            elif v in d:
                d[v] += [(k, v)]
            else:
                d[k] = [(k, v)]

            list_aux[v] = k

    return d

def dist_matrix(n_nodes: int, w_max=10)-> np.ndarray:
    """
        Nombre:
            dist_matrix

        Descripción:
            Función que genera una matriz cuadrada de distancias.

        Argumentos:
            - n_nodes: numero de nodos de la matriz.
            - w_max: valor máximo de cada nodo.
        Retorno:
            - Matriz de distancias.
    """

    #generamos la matriz aleatoria
    M = np.random.randint(1, w_max, size=(n_nodes, n_nodes))
    #hacemos que la matriz sea simetria respecto a la diagonal
    for i in range(n_nodes):
        for j in range(n_nodes):
            M[j][i] = M[i][j]

    #hacemos que la diagonal sea 0
    np.fill_diagonal(M,0)

    return M

def greedy_tsp(dist_m: np.ndarray, node_ini=0)-> List:
    """
        Nombre:
            greedy_tsp
        
        Descripción:
            Función que buscará el circuito en la matriz, partiendo desde el nodo 
            inicial siguiendo las distancias más cortas.

        Argumentos:
            - dist_m: matriz de distancias.
            - node_ini: nodo en el que empieza el circuito.

        Retorno:
            - La lista correspondiente al circuito.
    """
    num_nodes = dist_m.shape[0]

    lista = [node_ini]

    while len(lista) < num_nodes:
        current_node = lista[-1]

        not_used_nodes = list(np.argsort(dist_m[current_node]))

        for i in not_used_nodes:
            if i not in lista:
                lista.append(i)
                break

    return lista + [node_ini]

def len_circuit(circuit: List, dist_m: np.ndarray) -> int:
    """
        Nombre:
            len_circuit
        
        Descripción:
            Función que calculará la longitud total del circuito.

        Argumentos:
            - circuit: circuito.
            - dist_m: matriz de distancias.

        Retorno:
            - Longitud total.
    """
    pos_ini = circuit[0]
    suma = 0

    for i in circuit:
        if i == pos_ini:
            pass
        else:
            pos_next = i
            suma += dist_m[pos_ini][pos_next]
            pos_ini = pos_next       

    return suma

def repeated_greedy_tsp(dist_m: np.ndarray) -> List:
    """
        Nombre:
            repeated_greedy_tsp

        Descripción:
            Función que devolverá el circuito más corto. Es decir, tras buscar
            la distancia de todas las opciones, devolverá el que tenga el circuito
            con menor distancia recorrida.

        Argumentos:
            - dist_m: matriz de distancias.

        Retorno:
            - Circuito más corto.
    """
    tam = dist_m.shape[0]
    lista_aux = []

    for i in range(tam):
        lista_aux.append(len_circuit(greedy_tsp(dist_m, i), dist_m))

    indiceMin = np.argmin(lista_aux)
            
    return greedy_tsp(dist_m, indiceMin)

def exhaustive_tsp(dist_m: np.ndarray) -> List:
    """
        Nombre:
            exhaustive_tsp
        
        Descripción:
            Utilizando la librería de itertools, comprobaremos todas las permutaciones posibles y veremos
            cuál tiene la menor longitud de camino.

        Argumentos:
            - dist_m: matriz de distancias.

        Retorno:
            - List: array del circuito con menor longitud.
    """
    tam = dist_m.shape[0]
    lista_aux = []
    tam_aux = None

    # itertools.permutations devuelve TODAS las iteraciones posibles entre lo pasado por parámetro, en este caso un array de índices.
    for i in itertools.permutations(range(tam)):
        aux2 = np.append(i, i[0])

        tam2 = len_circuit(aux2, dist_m)

        if tam_aux == None or tam_aux > tam2:
            tam_aux = tam2
            lista_aux = aux2
    
    return lista_aux

if __name__ == "__main__":
    arb = [1, -2, 1, -5, 3, 4, 5, 6]
    print(find(0, arb))
    print(find(7, arb))
    print(f"El diccionario de los subconjuntos es: {cd_2_dict(arb)}")

    union(4, 6, arb)
    print(f"El padre de todos es: {find(0, arb)}")
    print(find(7, arb))

    l = [(0, 1), (0, 3), (1, 2), (1, 4), (3, 4), (4, 5), (4, 6), (5, 6)]

    print(f"El diccionario creado es: {ccs(7, l)}")

    print("------------------------------------")
    M = dist_matrix(6)
    for i in range(len(M)):
        print(M[i])

    L = greedy_tsp(M)
    print(L)
    print(len_circuit(L, M))
    print(repeated_greedy_tsp(M))
