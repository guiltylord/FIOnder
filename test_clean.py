import re
word = 'Уsales&ManagementЩ'
print(f'Оригинал: {word}')
clean = re.sub(r'^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$', '', word)
print(f'После очистки: {clean}')
