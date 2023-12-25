// Displays the image once it's selected
function displayImage(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
        const img = document.createElement('img');
        img.src = e.target.result;
        img.alt = "Dropped Image";
        img.width = 200;
        img.height = 200;
        // Append the image to the drop zone
        const dropZone = document.getElementById('drop-zone');
        dropZone.innerHTML = ''; // Clear previous content
        dropZone.appendChild(img);

        // Add the confirmed class after the image is appended
        img.classList.add('confirmed');
    };
    reader.readAsDataURL(file);
}


function handleDrop(event) {
    event.preventDefault();
    event.target.classList.remove('hover');

    const files = event.dataTransfer.files;
    if (files.length > 0) {
        handleFiles(files);

        // Update the file input with the dropped file
        const fileInput = document.getElementById('file-upload');
        fileInput.files = files;
    }
}

function handleFiles(files) {
    const file = files[0];
    displayImage(file); // Display dropped image
}




function handleDropZoneClick() {
    document.getElementById('file-upload').value = null; // Clear previously selected file
    document.getElementById('file-upload').click();
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFiles([file]);
    }
}

document.getElementById('upload-form').addEventListener('submit', function (event) {
    event.preventDefault();
    const files = document.getElementById('file-upload').files;
    if (files.length > 0) {
        handleFiles(files);
    }
});



function handleHover() {
    document.getElementById('drop-zone').classList.add('hover');
}

function handleOut() {
    document.getElementById('drop-zone').classList.remove('hover');
}

function handleDragOver(event) {
    event.preventDefault();
}

function handleDragEnter(event) {
    event.target.classList.add('hover');
}

function handleDragLeave(event) {
    event.target.classList.remove('hover');
}




