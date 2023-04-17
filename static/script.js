function makePrediction() {
    const textInput = document.getElementById("text-input").value;
    const data = { text: textInput };
    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(results => {
        const resultsDiv = document.getElementById("prediction-results");
        resultsDiv.innerHTML = "";
        results.forEach(result => {
            const predictedClass = result.predicted_class;
            const probability = result.probability;
            const text = result.text;
            const resultDiv = document.createElement("div");
            resultDiv.classList.add("result");
            resultDiv.innerHTML = `<p class="predicted-class">The predicted class is: ${predictedClass} <span class="probability">(${probability})</span></p><p class="text"><b>The Text Is:</b> ${text}</p><hr>`;
            resultsDiv.appendChild(resultDiv);
        });
        const predictionResults = document.querySelector('#prediction-results');
        predictionResults.classList.add('show');
    })
    .catch(error => console.error(error));
}
