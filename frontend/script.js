async function predict(temperature, humidity, hour) {
  const response = await fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      temperature: temperature,
      humidity: humidity,
      hour: hour
    })
  });

  if (!response.ok) {
    throw new Error("Backend error: " + response.status);
  }

  const data = await response.json();
  return data.prediction;
}