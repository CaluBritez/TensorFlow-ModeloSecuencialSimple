let model
let bandera = false
async function entrenar() {
    // Create a simple model.
    model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
  
    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
  
    // Generate some synthetic data for training. (y = 2x - 1)
    const xs = tf.tensor2d([-6,-3, -1, 0, 2, 4, 6, 8, 10], [9, 1]);
    const ys = tf.tensor2d([-6, 0, 4, 6, 10, 14, 18, 22, 26], [9, 1]);
  
    // Train the model using the data.
    await model.fit(xs, ys, {epochs: 500});
    
    document.getElementById('micro-out-div').innerText = "Entrenamiento Finalizado - Ahora puedes ingresar un valor";
    bandera = true
  }

  async function predecir() {
    if(bandera == true){
        const valorInput = parseFloat(document.getElementById('x').value);
        if(isNaN(valorInput)){
            document.getElementById('micro-out-div').innerText ="Debes ingresar un valor"
        }else{
            const resultado = model.predict(tf.tensor2d([valorInput], [1, 1])).dataSync();
            const resultadoRedondeado = resultado[0].toFixed(2);
            document.getElementById('micro-out-div').innerText =`El valor de y para x = ${valorInput} es de: ${resultadoRedondeado}`    	
            }    
    }else{
        document.getElementById('micro-out-div').innerText ="Primero debes entrenar el modelo"
    }
  }
  