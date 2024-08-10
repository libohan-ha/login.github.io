document.addEventListener('DOMContentLoaded', function() {
    const todoText = document.getElementById('todoText');
    const popupBoxes = document.getElementById('popup-boxes');
    const alertBox = document.getElementById('alert-box');
    const clockElement = document.getElementById('clock');

    let todos = [];
    let lastClickIndex = null;
    let lastClickTime = 0;
    let showAlert = false;

    function updateClock() {
        const now = new Date();
        const month = String(now.getMonth() + 1).padStart(2, '0'); // 月份从0开始，所以加1
        const day = String(now.getDate()).padStart(2, '0');
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');

        clockElement.textContent = `${month}月${day}日 ${hours}:${minutes}:${seconds}`;
    }

    function addTodo() {
        const text = todoText.value.trim();
        if (text !== '') {
            if (todos.length >= 6) {
                showAlert = true;
                alertBox.classList.add('show');
                setTimeout(() => {
                    alertBox.classList.remove('show');
                    showAlert = false;
                }, 2000);
                return;
            }
            todos.push({ text: text, show: true, blue: false, top: `${23 + todos.length * 10}%` });
            todoText.value = '';
            renderTodos();
        }
    }

    function insertNewLine(event) {
        if (event.key === 'Enter' && event.shiftKey) {
            todoText.value += '\n';
        }
    }

    function toggleColor(index) {
        todos[index].blue = !todos[index].blue;
        renderTodos();
    }

    function handleClick(index) {
        const now = Date.now();
        if (lastClickIndex === index && (now - lastClickTime) < 300) {
            // 连续点击两下，删除方框
            todos.splice(index, 1);
            updatePopupPositions();
        } else {
            // 单击，切换颜色
            toggleColor(index);
        }
        lastClickIndex = index;
        lastClickTime = now;
    }

    function updatePopupPositions() {
        todos.forEach((todo, index) => {
            todo.top = `${23 + index * 10}%`;
        });
        renderTodos();
    }

    function renderTodos() {
        popupBoxes.innerHTML = '';
        todos.forEach((todo, index) => {
            const popupBox = document.createElement('div');
            popupBox.className = `popup-box ${todo.show ? 'show' : ''} ${todo.blue ? 'blue' : ''}`;
            popupBox.style.top = todo.top;
            popupBox.onclick = () => handleClick(index);

            const content = document.createElement('div');
            content.className = 'popup-box-content';
            content.textContent = todo.text;

            popupBox.appendChild(content);
            popupBoxes.appendChild(popupBox);
        });
    }

    todoText.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            addTodo();
        } else if (event.key === 'Enter' && event.shiftKey) {
            event.preventDefault();
            insertNewLine(event);
        }
    });

    updateClock();
    setInterval(updateClock, 1000);
});
