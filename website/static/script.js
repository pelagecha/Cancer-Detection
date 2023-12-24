// script.js

function handleDragOver(event) {
    event.preventDefault();
}

function handleDragEnter(event) {
    event.target.classList.add('hover');
}

function handleDragLeave(event) {
    event.target.classList.remove('hover');
}

function handleDrop(event) {
    event.preventDefault();
    event.target.classList.remove('hover');

    const files = event.dataTransfer.files;
    if (files.length > 0) {
        handleFiles(files);
    }
}

function handleFiles(files) {
    const file = files[0];
    const reader = new FileReader();
    reader.onload = function (e) {
        const img = document.createElement('img');
        img.src = e.target.result;
        img.alt = "Dropped Image";
        img.width = 200; // Set the width as needed
        img.height = 200; // Set the height as needed
        document.body.appendChild(img);
    };
    reader.readAsDataURL(file);
}
