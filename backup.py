# backup.py
import os
import shutil
import zipfile
from datetime import datetime

def create_backup():
    """Создает резервную копию данных системы."""
    try:
        # Создаем папку backup если её нет
        if not os.path.exists("backup"):
            os.makedirs("backup")
        
        # Формируем имя архива с датой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup/skolovo_backup_{timestamp}"
        zip_name = f"{backup_name}.zip"
        
        # Создаем ZIP архив
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Добавляем chroma_db
            if os.path.exists("chroma_db"):
                for root, dirs, files in os.walk("chroma_db"):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_path = os.path.relpath(file_path, ".")
                        zipf.write(file_path, arc_path)
            
            # Добавляем data
            if os.path.exists("data"):
                for root, dirs, files in os.walk("data"):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_path = os.path.relpath(file_path, ".")
                        zipf.write(file_path, arc_path)
            
            # Добавляем YAML файлы
            yaml_files = ["corrections.yaml", "users.yaml"]
            for yaml_file in yaml_files:
                if os.path.exists(yaml_file):
                    zipf.write(yaml_file)
            
            # Добавляем logs
            if os.path.exists("logs"):
                for root, dirs, files in os.walk("logs"):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_path = os.path.relpath(file_path, ".")
                        zipf.write(file_path, arc_path)
        
        print(f"✅ Резервная копия создана: {zip_name}")
        
        # Удаляем старые резервные копии (оставляем только последние 7)
        cleanup_old_backups()
        
        return zip_name
    except Exception as e:
        print(f"❌ Ошибка при создании резервной копии: {e}")
        return None

def cleanup_old_backups():
    """Удаляет старые резервные копии, оставляя только последние 7."""
    try:
        backup_files = [f for f in os.listdir("backup") if f.startswith("skolovo_backup_") and f.endswith(".zip")]
        backup_files.sort(reverse=True)  # Сортируем по дате, новые первыми
        
        # Удаляем все кроме последних 7
        for old_backup in backup_files[7:]:
            os.remove(os.path.join("backup", old_backup))
            print(f"🗑️ Удалена старая резервная копия: {old_backup}")
    except Exception as e:
        print(f"❌ Ошибка при очистке старых резервных копий: {e}")

def restore_backup(backup_file):
    """Восстанавливает данные из резервной копии."""
    try:
        if not os.path.exists(backup_file):
            print(f"❌ Файл резервной копии не найден: {backup_file}")
            return False
        
        # Распаковываем архив
        with zipfile.ZipFile(backup_file, 'r') as zipf:
            zipf.extractall(".")
        
        print(f"✅ Восстановление из резервной копии выполнено: {backup_file}")
        return True
    except Exception as e:
        print(f"❌ Ошибка при восстановлении из резервной копии: {e}")
        return False

def list_backups():
    """Возвращает список доступных резервных копий."""
    try:
        if not os.path.exists("backup"):
            return []
        
        backup_files = [f for f in os.listdir("backup") if f.startswith("skolovo_backup_") and f.endswith(".zip")]
        backup_files.sort(reverse=True)  # Сортируем по дате, новые первыми
        return backup_files
    except Exception as e:
        print(f"❌ Ошибка при получении списка резервных копий: {e}")
        return []

if __name__ == "__main__":
    print("🔧 Утилита резервного копирования системы 'Консультант по Сколково'")
    print("1. Создать резервную копию")
    print("2. Просмотреть доступные резервные копии")
    print("3. Восстановить из резервной копии")
    
    choice = input("Выберите действие (1-3): ")
    
    if choice == "1":
        create_backup()
    elif choice == "2":
        backups = list_backups()
        if backups:
            print("\nДоступные резервные копии:")
            for i, backup in enumerate(backups, 1):
                print(f"{i}. {backup}")
        else:
            print("Резервные копии не найдены")
    elif choice == "3":
        backups = list_backups()
        if backups:
            print("\nДоступные резервные копии:")
            for i, backup in enumerate(backups, 1):
                print(f"{i}. {backup}")
            
            try:
                index = int(input("Выберите номер резервной копии для восстановления: ")) - 1
                if 0 <= index < len(backups):
                    restore_backup(os.path.join("backup", backups[index]))
                else:
                    print("Неверный номер")
            except ValueError:
                print("Неверный ввод")
        else:
            print("Резервные копии не найдены")
    else:
        print("Неверный выбор")