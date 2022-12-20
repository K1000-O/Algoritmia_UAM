"""
Fichero en el que se implementará las funciones necesarias para la realización del exámen.

Martes 20 de diciembre de 2022

Autor:

    Camilo Jené Conde ==> camilo.jenec@estudiante.uam.es

"""
import numpy as np

def heapsort(h: np.ndarray):
    """
    APARTADO A:
        Nos piden crear el algoritmo de ordenación heapsort.
    """
    # Se realiza el heapify de la lista entera.
    for i in range(np.size(h)//2, -1, -1):
        heapify(h, np.size(h)-1, i)

    # Se intercambia el primer elemento con el último, siendo éste el i recorriendo la lista en reversa.
    for i in range(np.size(h) - 1, -1, -1):
        h[i], h[0] = h[0], h[i]
        heapify(h, i-1, 0)

    return h

def heapify(h: list, n: int, i: int):
    """
    Función auxiliar utilizada para ordenar con un heapsort.
    """
    max = i

    # Si el hijo izquierdo es mayor que la raíz.
    if 2*i + 1 <= n and h[2*i + 1] > h[max]:
        max = 2*i + 1

    # Si el hijo derecho es mayor que la raíz
    if 2*i + 2 <= n and h[2*i + 2] > h[max]:
        max = 2*i + 2

    if max != i:
        h[i], h[max] = h[max], h[i]
        heapify(h, n, max)

def max_heapify(h: np.ndarray, i: int) -> np.ndarray:
    """
    Función que ordena de mayoyr a menor. Hace lo contrario que min_heapify().
    """
    while not (i*2)+2> h.size:
        k = i
        if h[i] > h[2*i+1]:
            k = 2*i+1
        if 2*i+2 < h.size and h[i] > h[2*i+2] and h[2*i+2] < h[k]:
            k = 2*i+2
        if k > i:
            h[i], h[k] = h[k], h[i]
            i = k
        else:
            return

def create_max_heap(h: np.ndarray) -> np.ndarray:
    """
    Función que crea el max_heap similar a cómo lo tenemos en la práctica con el create_min_heap.
    """
    for i in range(len(h) // 2 - 1, -1, -1):
        max_heapify(h, i)
    return h

def max_extract(h: np.ndarray) -> tuple((int, np.ndarray)):
    """
    Función que extrae un elemento del heap. Similar a pq_extract() realizado en la práctica.
    """
    if h.size == 0:
        return
    elif h.size == 1:
        return (h[0], np.empty(0, dtype=int))

    m = h[0]

    h_aux = np.append([h[-1]], h[1:-1])
    max_heapify(h_aux, 0)

    return (m, h_aux)



def heapsort_rev(h: np.array) -> np.array:
    """
    ABARTADO B:
        Nos piden realizar el algoritmo de ordenación de forma reversa. Ordenamos de mayor a menor.
    """
    for i in range(len(h)//2, -1, -1):
        heapify_rev(h, len(h)-1, i)

    for i in range(len(h)-1, -1, -1):
        h[i], h[0] = h[0], h[i]
        heapify_rev(h, i-1, 0)
    
    return h
    
def heapify_rev(h: np.array, n: int, i: int):
    """
    Función auxiliar utilizada para ordenar el heapsort_rev
    """
    max = i

    # Comprobamos que el hijo izquierdo es menor que la raíz.
    if 2*i + 1 <= n and h[2*i + 1] < h[max]:
        max = 2*i + 1

    # Comprobamos que el hijo derecho es menor que la raíz.
    if 2*i+2 <= n and h[2*i + 2] < h[max]:
        max = 2*i+2
        
    if max != i:
        h[i], h[max] = h[max], h[i]
        heapify_rev(h, n, max)
