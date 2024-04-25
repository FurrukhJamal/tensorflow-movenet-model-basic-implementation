import logo from "./logo.svg";
import "./App.css";
import { useState, useEffect, useRef } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
// Register one of the TF.js backends.
import "@tensorflow/tfjs-backend-webgl";
// import '@tensorflow/tfjs-backend-wasm';

import testImage from "./testImage.jpg";

function App() {
  const [model, setModel] = useState(null);
  const imageRef = useRef(null);
  const [poses, setPoses] = useState([]);

  useEffect(() => {
    (async () => {
      await tf.ready();
      tf.backend("webgpu");

      const detectorConfig = {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        enableTracking: true,
        trackerType: poseDetection.TrackerType.BoundingBox,
      };
      const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        detectorConfig
      );
      setModel(detector);
      // console.log("detector : ", detector);
    })();
  }, []);

  useEffect(() => {
    // console.log("model : ", model);
    if (model) {
      (async () => {
        let exampleInput = tf.zeros([1, 192, 192, 3]);

        const img = new Image();
        img.onload = function () {
          const width = this.naturalWidth;
          const height = this.naturalHeight;

          console.log(`width : ${width} , height : ${height}`);
        };

        img.src = testImage;

        const p = await model.estimatePoses(imageRef.current);
        // console.log("poses are : ", poses);
        setPoses(p[0].keypoints);
      })();
    }
  }, [model]);

  useEffect(() => {
    console.log("poses : ", poses);
  }, [poses]);

  return (
    <div>
      <h1>TensorFLow JS Movenet Model</h1>
      {model ? <p>Model Loaded</p> : <p>Model is loading ...</p>}
      <div className="imageContainer">
        <img ref={imageRef} src={testImage} width="640" height="360" />
        {poses.map((pose, index) => (
          <div
            key={index}
            className="posePointDot"
            style={{
              top: parseFloat(pose["y"]) - 2,
              left: parseFloat(pose["x"]) - 2,
            }}
          ></div>
        ))}
      </div>
    </div>
  );
}

export default App;
