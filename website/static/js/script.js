let isDropped = false;

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

function handleDrop(event) {
    event.preventDefault();
    event.target.classList.remove('hover');

    const files = event.dataTransfer.files;
    if (files.length > 0) {
        handleFiles(files);
        uploadFiles(files);
    }
}

function handleFiles(files) {
    const file = files[0];
    displayImage(file); // Display dropped image
}

function displayImage(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
        const img = document.createElement('img');
        img.src = e.target.result;
        img.alt = "Dropped Image";
        img.width = 200;
        img.height = 200;
        const dropZone = document.getElementById('drop-zone');
        dropZone.innerHTML = '';
        dropZone.appendChild(img);
    };
    reader.readAsDataURL(file);
}



function handleDropZoneClick() {
    isDropped = false;
    document.getElementById('file-upload').value = null; // Clear previously selected file
    document.getElementById('file-upload').click();
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        isDropped = false;
        handleFiles([file]);
    }
}

document.getElementById('file-upload').addEventListener('change', handleFileSelect);

document.getElementById('upload-form').addEventListener('submit', function (event) {
    event.preventDefault();
    const files = document.getElementById('file-upload').files;
    if (files.length > 0) {
        handleFiles(files);
    }
});

function handleUpload() {
    const files = document.getElementById('file-upload').files;
    if (files.length > 0) {
        handleFiles(files);
        uploadFiles(files);
    }
}

function uploadFiles(files) {
    const formData = new FormData();
    formData.append('file', files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log('File uploaded successfully!');
        // If needed, update the displayed image here after successful upload
    })
    .catch(error => {
        console.error('Error uploading file:', error);
    });
}

