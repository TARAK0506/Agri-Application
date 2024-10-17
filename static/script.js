document.getElementById('recommendation-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    const formData = new FormData(this);
    const response = await fetch('/recommend', {
        method: 'POST',
        body: JSON.stringify(Object.fromEntries(formData)),
        headers: {
            'Content-Type': 'application/json'
        }
    });
    const result = await response.json();
    document.getElementById('recommendation-result').innerText = `Recommended Crop: ${result.recommended_crop}`;
});

document.getElementById('disease-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    const fileInput = document.getElementById('file');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    // Handle error response
    if (response.ok) {
        document.getElementById('disease-result').innerText = `Predicted Disease: ${result.prediction}`;
    } else {
        document.getElementById('disease-result').innerText = `Error: ${result.error}`;
    }
});