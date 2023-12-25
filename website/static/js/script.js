// Displays the image once it's selected
// Displays the image when Upload is pressed
function displaySelectedImage() {
    const fileInput = document.getElementById('file-upload');
    const files = fileInput.files;

    if (files.length > 0) {
        const file = files[0];
        displayImage(file); // Display dropped image
    }
}

// The modified displayImage function
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

        // Delay adding the confirmed class to ensure transition
        setTimeout(function() {
            img.classList.add('confirmed');
        }, 50); // Adjust delay as needed
    };
    reader.readAsDataURL(file);
}

// Function to handle Upload button click
function handleUploadButtonClick() {
    displaySelectedImage();
}

// Event listener for Upload button click
document.getElementById('upload-button').addEventListener('click', function (event) {
    event.preventDefault();
    handleUploadButtonClick();
});




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

function makePrediction() {
    console.log("called function");
    let butt = document.getElementById("prediction-button");
    let text = document.getElementById("prediction-text"); 
    let uploadButton = document.getElementById("new-upload-button");

    butt.style.display = "none";
    text.style.display = "block";
    uploadButton.style.display = "block";

    fetch('/prediction')
        .then(response => response.json())
        .then(data => {
            text.innerText = data.pred;
        })
}


