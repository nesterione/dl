#!/usr/bin/env python3
import random
import operator
import sys

class MatrixError(BaseException):
    """ Класс исключения для матриц """
    pass

class Matrix(object):
    """Простой класс матрицы в Python
    Основные операции линейной алгебры реализованы
    путем перегрузки операторов 
    """
    def __init__(self, n, m, init=True):
        """Конструктор

        #Аргументы:
            n    :  int, число строк
            m    :  int, число столбцов 
            init :  (необязательный параметр), логический.
                    если False, то создается пустой массив
        """
        if init:
            # создаем массив нулей
            self.array = [[0]*m for x in range(n)]
        else:
            self.array = []

        self.n = n
        self.m = m

    def __getitem__(self, idx):
        """Перегрузка оператора получения элемента массива 
        """
        # проверяем, если индекс - это список индексов
        if isinstance(idx, tuple): 
            if len(idx) == 2: 
                return self.array[idx[0]][idx[1]]
            else:
                # у матрицы есть только строки и столбцы
                raise MatrixError("Matrix has only two shapes!")
        else:
            return self.array[idx]

    def __setitem__(self, idx, item):
        """Перегрузка оператора присваивания 
        """
        # проверяем, если индекс - это список индексов
        if isinstance(idx, tuple):
            if len(idx) == 2: 
                self.array[idx[0]][idx[1]] = item
            else:
                # у матрицы есть только строки и столбцы
                raise MatrixError("Matrix has only two shapes!")
        else:
            self.array[idx] = item

    def __str__(self):
        """Переопределяем метод вывода матрицы в консоль
        """
        s='\n'.join([' '.join([str(item) for item in row]) for row in self.array])
        return s + '\n'

    @property
    def rank(self):
        """Получить число строк и столбцов
        """
        return (self.n, self.m)


    def __eq__(self, mat):
        """ Проверка на равенство """

        return (mat.array == self.array)

    def transpose(self):
        """
        Транспонированное представление матрицы
        """
        ret = Matrix(self.m, self.n)
        for i in range(self.m):
            for j in range(self.n):
                ret.array[i][j] = self.array[j][i]
        return ret

    @staticmethod
    def _mat_sum(a, b, sign=False):
        if a.rank != b.rank:
            raise MatrixError("Trying to add matrixes of varying rank!")

        ret = Matrix(a.n, a.m, init=True)
        for i in range(a.n):
            for j in range(a.m):
                lvalue = a.array[i][j]
                rvalue = b.array[i][j]
                ret.array[i][j] = lvalue + rvalue if not sign else lvalue - rvalue
        return ret

    def __add__(self, mat):
        """
        Переопределение операции сложения "+" для матриц
        """
        return Matrix._mat_sum(self, mat)

    def __sub__(self, mat):
        """
        Переопределение операции вычитания "-" для матриц
        """
        return Matrix._mat_sum(self, mat, sign=True)

    def __mul__(self, mat):
        """Произведение Адамара или поточечное умножение"""
        mulmat = Matrix(self.n, self.m) # результирующая матрица

        # если второй аргумент - число, то 
        # просто умножить каждый элемент на это число
        if isinstance(mat, int) or isinstance(mat, float):
            for i in range(self.n):
                for j in range(self.m):
                    mulmat[i][j] = self.array[i][j]*mat
            return mulmat
        else:
            # для поточечного перемножения матриц  
            # их размерности должны быть одинаковыми
            if (self.n != mat.n or self.m != mat.m):
                raise MatrixError("Matrices cannot be multipled!")
                
            for i in range(self.n):
                for j in range(self.m):
                    mulmat[i][j] = self.array[i][j]*mat[i][j]
            return mulmat

    def dot(self, mat):
        """ Матричное умножение """
        
        # для перемножения матриц число столбцов одной 
        # должно равняться числу строк в другой
        if (self.m != mat.n):
            raise MatrixError("Matrices cannot be multipled!")

        depth = self.m
        ret = Matrix(self.n, mat.m)

        for i in range(ret.n):
            for j in range(ret.m):
                ret.array[i][j] = sum(
                        [self.array[i][p] * mat[p][j] for p in range(depth)]
                )

        return ret

    @classmethod
    def _makeMatrix(cls, array):
        """Переопределение конструктора
        """
        n = len(array)
        m = len(array[0])
        # Validity check
        if any([len(row) != m for row in array[1:]]):
            raise MatrixError("inconsistent row length")
        mat = Matrix(n,m, init=False)
        mat.array = array

        return mat

    @classmethod
    def fromList(cls, listoflists):
        """ Создание матрицы напрямую из списка """

        # E.g: Matrix.fromList([[1 2 3], [4,5,6], [7,8,9]])

        array = listoflists[:]
        return cls._makeMatrix(array)

    @classmethod
    def makeId(cls, n):
        """ Создать единичную матрицу размера (nxn) """

        array = [[0]*n for x in range(n)]
        idx = 0
        for row in array:
            row[idx] = 1
            idx += 1

        return cls.fromList(array)


if __name__ == "__main__":
    pass
