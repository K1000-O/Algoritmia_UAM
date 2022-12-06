"""
Fichero en el que introducimos los métodos necesarios para la correcta solución de los ejercicios propuestos por la práctica 1.

Autores:    
    
    Alejandro Raúl Hurtado Fuertes
    
    Camilo Jené Conde
"""
from typing import List
import numpy as np
from typing import Callable
from typing import Tuple

#   APARTADO 1.A
def matrix_multiplication(m_1: np.ndarray, m_2:np.ndarray):
    """ 
    Función que va a multiplicar ambas arrays recibidas cómo parámetros.

    Se devolverá un nuevo array con la solución.
    """
    n_rows, n_interm, n_columns = \
        m_1.shape[0], m_2.shape[0], m_2.shape[1]
    
    m_product = np.zeros( (n_rows, n_columns) )
    
    for p in range(n_rows):
        for q in range(n_columns):
            for r in range(n_interm):
                m_product[p, q] += m_1[p, r] * m_2[r, q]
        
    return m_product


#   APARTADO 1.B
def rec_bb(t:List, f:int, l:int, key:int) -> int:
    """
    Búsqueda binaria recursiva.

    Algoritmo:

        Ir partiendo en mitades el array hasta encontrar la key solicitada.
        Si no se encuentra, llamar de nuevo a la función con los nuevos valores.
    """
    if f > l:
        return None
    if f == l:
        if key == t[f]:
            return f
        else:
            return None

    mid = (f + l) // 2


    if key == t[mid]:
        return mid
    elif key < t[mid]:
        return rec_bb(t, f, mid-1, key)
    else:
        return rec_bb(t, mid+1, l, key)

def bb(t:List, f:int, l:int, key:int) -> int:
    """
    Búsqueda binaria iterativa.

    Algoritmo:

        Ir partiendo en mitades el array hasta encontrar la key solicitada.
        Si no se encuentra, cambiar las variables para que en la siguiente
        ejecución del bucle se realice con la nueva parte del array.
    """
    while(True):
        if f > l:
            return None
        if f == l:
            if key == t[f]:
                return f
            else:
                return None

        mid = (f + l) // 2


        if key == t[mid]:
            return mid
        elif key < t[mid]:
            l = mid - 1
        else:
            f = mid + 1

def min_heapify(h: np.ndarray, i: int):
    """
    Función encargada de realizar la ordenación del heap.

    Asegura que el heap resuelto sea un MIN HEAP.
    """
    while 2*i + 1 < len(h):
        n_i = i

        if h[i] > h[2*i+1]:
            n_i = 2*i+1

        if 2*i+2 < len(h) and h[i] > h[2*i+2] and h[2*i+2] < h[n_i]:
            n_i = 2*i+2

        if n_i > i:
            h[i], h[n_i] = h[n_i], h[i]
            i = n_i
        else:
            return h

    return h


def insert_min_heap(h: np.ndarray, k: int) -> np.ndarray:
    """
    Método que se encarga de introducir un elemento en el min heap.

    Lo inserta al final del array y lo ordena.
    """
    if h is None:
        h = np.array([k])
        return h
    else:
        h = np.append(h, k) # Añade al final del array la nueva key.

        tam = len(h) - 1

        while tam >= 1 and h[(tam - 1) // 2] > h[tam]:
            h[(tam - 1) // 2], h[tam] = h[tam], h[(tam - 1) // 2] # Intercambiamos los valores.
            tam = (tam - 1) // 2

        return h 

def create_min_heap(h: np.ndarray):
    """
    Seter del min heap.
    """
    h = None
    return h

#   APARTADO 2.B
def pq_ini():
    """
    Seter de la cola.
    """
    h = None
    return h

def pq_insert(h: np.ndarray, k: int) -> np.ndarray:
    """
    Función que inserta un elemento en la cola. Realizando la ordenación correspondiente.
    """
    return insert_min_heap(h, k)

def pq_remove(h: np.ndarray)-> Tuple[int, np.ndarray]:
    """
    Función que elimina el primer elemento del array. Asegura que deja la cola como un minHeap.
    """
    if h is None:
        return
    elif h.size == 1:
        return (h[0], pq_ini())

    m = h[0]
    h = np.delete(h, 0)

    h = min_heapify(h, 0)

    return (m, h)

def select_min_heap(h: np.ndarray, k: int)-> int:
    """
    Función creada para solventar el problema de selección.
    """
    if np.size(h)<k:
        return

    pq = pq_ini()

    for i in range(np.size(h)):
        pq = pq_insert(pq, -1*h[i])

    while np.size(pq)>=k:
        if pq is None:
            break
        else:
            (value, pq) = pq_remove(pq)

    return -1*value