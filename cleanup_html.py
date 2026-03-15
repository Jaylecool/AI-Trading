file_path = r'c:\Users\Admin\Documents\AI Trading\dashboard_trade_history.html'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the first occurrence of </html>
last_index = content.find('</html>')
if last_index != -1:
    # Keep only up to and including </html>
    new_content = content[:last_index + 7]
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('File cleaned successfully')
    print(f'Removed {len(content) - len(new_content)} characters')
else:
    print('</html> tag not found')
