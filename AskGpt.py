#!/usr/bin/env python3
import os

# Настройки
MAX_CHARS = 300000  # можно менять
OUTPUT_PREFIX = "message_"
FINISH_INSTRUCTION = "ГПТ, Я ЕЩЕ НЕ ЗАКОНЧИЛ  - ПРОСТО КОРОТКО ОТВЕТЬ ОК И ВСЕ!!!"
BASE_DIR = os.getcwd()  # корневая папка — та, где запущен скрипт

# Список файлов для игнорирования, указывайте через запятую (например: "file1.txt, file2.log")
USER_IGNORE_FILES = "go.mod, go.sum, analyzer_wide, tz.md, Отчет об оптимизации параметров стратегий"  # можно менять

# Список папок для игнорирования, указывайте через запятую (например: "node_modules, .git")
USER_IGNORE_DIRS = "venv, logs, __pycache__, "  # можно менять

# Всегда игнорируем скрипт, сравнивая имя в нижнем регистре
SCRIPT_NAME_LOWER = os.path.basename(__file__).lower()

# Игнорируемые файлы из настроек тоже приводим к нижнему регистру
IGNORE_SET = set(name.strip().lower() for name in USER_IGNORE_FILES.split(",") if name.strip())

# Игнорируемые папки, приводим к нижнему регистру
IGNORE_DIRS_SET = set(name.strip().lower() for name in USER_IGNORE_DIRS.split(",") if name.strip())

def build_tree(root):
    """
    Строит строковое представление дерева папок.
    Файл скрипта (например, AskGpt.py) пропускается, независимо от регистра.
    Исправлено: базовая папка всегда имеет уровень 0, а вложенные папки начинаются с уровня 1.
    """
    tree_lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Игнорируем указанные папки
        dirnames[:] = [d for d in dirnames if d.lower() not in IGNORE_DIRS_SET]
        rel_path = os.path.relpath(dirpath, root)
        # Если rel_path равен '.', значит это базовая папка и глубина = 0,
        # иначе для вложенных папок прибавляем 1, чтобы базовая папка всегда была на уровне 0.
        depth = 0 if rel_path == '.' else rel_path.count(os.sep) + 1
        indent = "  " * depth
        tree_lines.append(f"{indent}[DIR] {os.path.basename(dirpath)}")
        for f in filenames:
            # Если имя файла (в нижнем регистре) совпадает с именем скрипта, пропускаем его
            if f.lower() == SCRIPT_NAME_LOWER:
                continue
            tree_lines.append(f"{indent}  [FILE] {f}")
    return "\n".join(tree_lines)

def get_file_content(filepath):
    """
    Возвращает содержимое файла, либо сообщение об ошибке.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"Ошибка чтения файла: {e}"

def get_all_file_paths(root):
    """
    Собирает пути ко всем файлам в BASE_DIR, за исключением файлов из IGNORE_SET и самого скрипта.
    Также игнорируются файлы, находящиеся в папках из IGNORE_DIRS_SET.
    """
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Игнорируем указанные папки
        dirnames[:] = [d for d in dirnames if d.lower() not in IGNORE_DIRS_SET]
        for f in filenames:
            if f.lower() == SCRIPT_NAME_LOWER or f.lower() in IGNORE_SET:
                continue
            file_paths.append(os.path.join(dirpath, f))
    return file_paths

def write_chunks(lines):
    """
    Разбивает общий текст на чанки, не превышающие MAX_CHARS символов.
    Если добавление строки превышает лимит, завершается чанк инструкцией FINISH_INSTRUCTION.
    """
    chunks = []
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) + 1 > MAX_CHARS:
            current_chunk += "\n" + FINISH_INSTRUCTION
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk = current_chunk + "\n" + line if current_chunk else line
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def main():
    all_lines = []
    explanation = (
        "сейчас я выгружу тебя сначала дерево файлов - а потом все файлы, "
        "возможно это займет больше одного сообщения, тогда просто скажи ОК, "
        "а затем уже задам вопросы - чтобы ты был в контексте кода"
    )
    all_lines.append(explanation)
    
    # Дерево папок (скрипт не выводится)
    tree = build_tree(BASE_DIR)
    all_lines.append("\n--- Структура папки ---\n")
    all_lines.append(tree)
    
    # Содержимое файлов (игнорируются файлы из IGNORE_SET и сам скрипт)
    all_lines.append("\n--- Содержимое файлов ---\n")
    file_paths = get_all_file_paths(BASE_DIR)
    for path in file_paths:
        rel_path = os.path.relpath(path, BASE_DIR)
        all_lines.append(f"\nФайл: {rel_path}\n{'-' * len('Файл: ' + rel_path)}")
        content = get_file_content(path)
        content_lines = content.splitlines()
        all_lines.extend(content_lines)
    
    # Разбиваем итоговый текст на чанки
    chunks = write_chunks(all_lines)
    for i, chunk in enumerate(chunks, 1):
        out_filename = f"{OUTPUT_PREFIX}{i}.txt"
        with open(out_filename, "w", encoding="utf-8") as out_file:
            out_file.write(chunk)
        print(f"Сохранено: {out_filename}")

if __name__ == "__main__":
    main()