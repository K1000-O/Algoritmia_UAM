"""
    Implementación de las funciones pedidadas para la solución de la práctica 3 de la asignatura de
    Practicas de Algoritmos y Estructuras de Datos Avanzadas.

    Autor: Alejandro Raúl Hurtado <alejandror.hurtado@estudiante.uam.es>

    Autor: Camilo Jené Conde <camilo.jenec@estudiante.uam.es>

    2022 EPS-UAM 
"""
import numpy as np
from typing import List, Tuple, Union

def split (t: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    #Nombre:
        split
    
    #Descripción:
        Función que realiza el split de un array. Situando el primer elemento entre sus elementos menores y mayores.

    #Argumentos:
        - t: array con elementos.
    
    #Return:
        - Tuple[np.ndarray, int, np.ndarray]: tupla con --> (array de elementos menores al pivote, pivote, elementos mayores).
    """
    p = t[0]
    lst = t[1:]

    elementosMenores = [x for x in lst if x <= p]
    elementosMayores = [x for x in lst if x > p]

    return (elementosMenores, p, elementosMayores)

def qsel(t: np.ndarray, k: int) -> Union[int, None]:
    """
    #Nombre:
        qsel
    
    #Descripción:
        Función que implementa el algoritmo QuickSelect de forma recursiva.

    #Argumentos:
        - t: array con los elementos.
        - k: índice del elemento que buscamos.

    #Return
        - int: elemento que se encuentra en el índice 'k' de la lista ordenada.
        - None: si el índice es mayor o menor que la tabla.
    """
    if k < 0 or k > np.size(t)-1:
        return None

    menores, pivote, mayores = split(t)
    tablaAux = menores + [pivote] + mayores

    if pivote == tablaAux[k]:
        return pivote

    if tablaAux[k] < pivote:
        return qsel(menores, k)
    else:
        return qsel(mayores, k-np.size(menores)-1)

def qsel_nr(t: np.ndarray, k: int) -> Union[int, None]:
    """
    #Nombre:
        qsel_nr
    
    #Descripción:
        Función que implementa el algoritmo QuickSelect de forma NO recursiva.

    #Argumentos:
        - t: array con los elementos.
        - k: índice del elemento que buscamos.

    #Return:
        - int: elemento que se encuentra en el índice 'k' de la lista ordenada.
        - None: si el índice es mayor o menor que la tabla.
    """
    t_actual = t

    while (True):
        if k < 0 or k > np.size(t)-1:
            return None

        menores, pivote, mayores = split(t_actual)
        
        tablaAux = menores + [pivote] + mayores

        if tablaAux[k] == pivote:
            return pivote 
        elif tablaAux[k] < pivote:
            t_actual = menores
        elif tablaAux[k] > pivote:
            k = k-np.size(menores)-1
            t_actual = mayores
           
def split_pivot(t: np.ndarray, mid: int) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    #Nombre:
        split_pivot
    
    #Descripción:
        Función que realiza el split de un array a partir de un elemento pasado por argumento.
        Situando el número cómo pivote y realizando las listas de elementos menores y mayores.

    #Argumentos:
        - t: array con elementos.
        - mid: elemento que utilizaremos como pivote.
    
    #Return:
        - Tuple[np.ndarray, int, np.ndarray]: tupla con --> (array de elementos menores al pivote, pivote, elementos mayores).
    """
    p = mid
    lst = t

    if mid in lst:
        lst = np.delete(lst, np.argmax(lst==mid))
    else:
        return -1

    elementosMenores = [x for x in lst if x <= p]
    elementosMayores = [x for x in lst if x > p]

    return (elementosMenores, p, elementosMayores)

def pivot5(t: np.ndarray) -> int:
    """
    #Nombre:
        pivot5
    
    #Descripción:
        Función que devuelve la mediana de las medianas de los elementos del array dividiéndolo en subconjuntos de 5.

    #Argumentos:
        - t: array con elementos.
    
    #Return:
        - int: elemento utilizado cómo pivote.
    """
    n_elementos_subconjunto = 5

    # num_sublistas = np.size(t) // n_elementos_subconjunto
    # sublistas = [t[i:i+5] for i in range(0, np.size(t), n_elementos_subconjunto)][:num_sublistas]
    
    sublistas = [t[i:i+5] for i in range(0, np.size(t), n_elementos_subconjunto)]

    medianas = [sorted(sub)[np.size(sub)//2] for sub in sublistas] # Realizamos el array de las medianas de cada sublista.

    return qsel5_nr(medianas, np.size(medianas)//2) # Devolvemos el pivote intermedio del array de medianas.

def qsel5_nr(t: np.ndarray, k: int) -> Union[int, None]:
    """
    #Nombre:
        qsel5_nr
    
    #Descripción:
        Función que implementa el algoritmo QuickSelect de forma NO recursiva con las funciones split_pivot y pivot5.

    #Argumentos:
        - t: array con los elementos.
        - k: índice del elemento que buscamos.

    #Return
        - int: elemento que se encuentra en el índice 'k' de la lista ordenada.
        - None: si el índice es mayor o menor que la tabla.
    """
    t_actual = t

    while (True):
        if k < 0 or k > np.size(t)-1:
            return None

        menores, pivote, mayores = split_pivot(t_actual, t_actual[k])
        
        tablaAux = menores + [pivote] + mayores

        if tablaAux[k] == pivote:
            return pivote 
        elif tablaAux[k] < pivote:
            t_actual = menores
        elif tablaAux[k] > pivote:
            k = k-np.size(menores)-1
            t_actual = mayores

def qsort_5(t: np.ndarray) -> np.ndarray:
    """
    #Nombre:
        qsort5
    
    #Descripción:
        Función que implementa el algoritmo de ordenación QuickSort de forma recursiva.

    #Argumentos:
        - t: array con los elementos.

    #Return
        - np.ndarray: array ordenada.
    """
    if np.size(t) <= 1:
        return t

    pivote = pivot5(t)

    menores, pivote_aux, mayores = split_pivot(t, pivote)

    return qsort_5(menores) + [pivote_aux] + qsort_5(mayores)
    

"""
    A PARTIR DE AQUÍ SE REALIZAN LAS FUNCIONES PARA LA PARTE II.
"""
def edit_distance(str_1: str, str_2: str) -> int:
    """
    #Nombre:
        edit_distance
    
    #Descripción:
        Función que devuelve el número de intercambios que necesitamos para llegar de una palabra a otra.

    #Argumentos:
        - str_1: primera palabra.
        - str_2: segunda palabra.

    #Return
        - int: entero con el número de intercambios entre una palabra y otra.
    """
    if len(str_1) > len(str_2):
        str_1, str_2 = str_2, str_1

    distances = range(len(str_1) + 1)

    for i2, c2 in enumerate(str_2):
        contador_ = [i2+1]

        for i1, c1 in enumerate(str_1):
            if c1 == c2:
                contador_.append(distances[i1])
            else:
                contador_.append(1 + min(distances[i1], distances[i1+1], contador_[-1]))
        
        distances = contador_

    return distances[-1]

def max_subsequence_length(str_1: str, str_2: str) -> int:
    distances = np.zeros((len(str_1), len(str_2)), dtype=int)

    for i in range(0, len(str_1)):
        for j in range(0, len(str_2)):
            if i == 0 or j == 0:
                continue
            elif str_1[i] == str_2[j]:
                distances[i][j] = 1 + distances[i-1][j-1]
            else:
                distances[i][j] = max([distances[i][j-1], distances[i-1][j]])

    return distances[len(str_1)-1][len(str_2)-1]

def max_common_subsequence(str_1: str, str_2: str)-> str:
    matrix = np.full((len(str_1), len(str_2)), "")

    for i in range(0, len(str_1)):
        for j in range(0, len(str_2)):
            if i == 0 or j == 0:
                continue
            elif str_1[i] == str_2[j]:
                matrix[i][j] = matrix[i-1][j-1] + str_1[i]
            elif len(matrix[i][j-1]) > len(matrix[i-1][j]):
                matrix[i][j] = matrix[i][j-1]
            else:
                matrix[i][j] = matrix[i-1][j]
    return matrix[len(str_1)-1]

def min_mult_matrix(l_dims: List[int])-> int:
    matrix = np.zeros((len(l_dims)-1, len(l_dims)-1), dtype=int)
    for dif in range(1, len(l_dims)-2):
        for start in range(len(l_dims)-2-dif):
            end = start + dif
            # start y end las matrices de las que se quiere hallar el numero de productos
            # x dimensiones: l_dims[x] l_dims[x+1]
            cost = -1
            for j in range(start, end):
                if cost == -1:
                    cost = matrix[start][j] + matrix[j+1][end] + l_dims[start] * l_dims[j+1] * l_dims[end+1]
                else:
                    cost = min([cost, matrix[start][j] + matrix[j+1][end] + l_dims[start] * l_dims[j+1] * l_dims[end+1]])
            matrix[start][end] = cost
    return matrix[0][len(l_dims)-2]

if __name__ == "__main__":
    t = [5, 3, 1, 4, 8, 6, 7, 2, 2, 5, 5, 5] 

    tabla = split(t)
    menores, pivote, mayores = split(t)

    print(menores)
    print(mayores)
    print(tabla)

    print("\n ------ PRUEBA DE QSEL ------")
    print(qsel(t, 0))
    print(qsel(t, 1))
    print(qsel(t, 2))
    print(qsel(t, 3))
    print(qsel(t, 4))
    print(qsel(t, 5))
    print(qsel(t, 6))
    print(qsel(t, 7))
    print(qsel(t, 8))
    print(qsel(t, 9))
    print(qsel(t, 10))
    print(qsel(t, 11))
    print(qsel(t, 12))
    print(qsel(t, 13))
    print(qsel(t, 14))

    print("\n ------ PRUEBA DE QSEL_NR ------")
    print(qsel_nr(t, 0))
    print(qsel_nr(t, 1))
    print(qsel_nr(t, 2))
    print(qsel_nr(t, 7))
    print(qsel_nr(t, 8))
    print(qsel_nr(t, 9))

    print("\n ------ PRUEBA DE pivot5 ------")
    print(pivot5(t))

    print("\n ------ PRUEBA DE qsort_5 ------")
    print(qsort_5(t))

    print("\n ------ PRUEBA DE edit_dist ------")
    print(edit_distance("biscuit", "suitcase"))

    print("\n ------ PRUEBA DE max_subsequence ------")
    print(max_subsequence_length("forrajes", "zarzajo"))
    print(max_common_subsequence("forraje", "zarzajo"))

    """
    """
        # preunejrnkafnadsnfinadsifj
    """
    """