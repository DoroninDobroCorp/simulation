#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль для запуска приложения.
"""

import sys
from PyQt5.QtWidgets import QApplication
from gui import MainWindow


def main():
    """
    Основная функция для запуска приложения.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 