'use client';

import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { AlertCircle, Download } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { HfInference } from '@huggingface/inference';
import { HolographicNeuralNetwork } from '@/lib/HolographicNeuralNetwork';

export default function HolographicChat() {
  const [chatHistory, setChatHistory] = useState<
    { type: string; text: string }[]
  >([]);
  const [inputText, setInputText] = useState('');
  const [hnn, setHnn] = useState<HolographicNeuralNetwork | null>(null);
  const [learnInput, setLearnInput] = useState('');
  const [learnResponse, setLearnResponse] = useState('');
  const [isLearning, setIsLearning] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [isLLMActive, setIsLLMActive] = useState(false);
  const [isLoadingLLM, setIsLoadingLLM] = useState(false);
  const [llmModel, setLLMModel] = useState<HfInference | null>(null);
  const sceneRef = useRef<HTMLDivElement | null>(null);
  const neuronMeshesRef = useRef<THREE.Mesh[]>([]);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  const smallDataset = [
    {
      question: 'What is the capital of France?',
      answer: 'The capital of France is Paris.',
    },
    {
      question: 'Who painted the Mona Lisa?',
      answer: 'Leonardo da Vinci painted the Mona Lisa.',
    },
    {
      question: 'What is the largest planet in the solar system?',
      answer: 'Jupiter is the largest planet in the solar system.',
    },
    {
      question: 'In which year did World War II begin?',
      answer: 'World War II began in 1939.',
    },
    {
      question: 'What is the most abundant chemical element in the universe?',
      answer: 'Hydrogen is the most abundant chemical element in the universe.',
    },
  ];

  useEffect(() => {
    const newHnn = new HolographicNeuralNetwork(1000, 100, 0.005);
    setHnn(newHnn);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / 2 / window.innerHeight,
      0.1,
      1000
    );
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth / 2, window.innerHeight);
    if (sceneRef.current) {
      sceneRef.current.appendChild(renderer.domElement);
    }
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    camera.position.z = 150;

    const geometry = new THREE.SphereGeometry(0.5, 32, 32);
    const material = new THREE.MeshStandardMaterial({
      color: 0x444444,
      emissive: 0x000000,
      emissiveIntensity: 1,
      toneMapped: false,
    });

    newHnn.neurons.forEach((neuron) => {
      const mesh = new THREE.Mesh(geometry, material.clone());
      mesh.position.copy(neuron.position);
      scene.add(mesh);
      neuronMeshesRef.current.push(mesh);
    });

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 0);
    scene.add(directionalLight);

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Initialize LLM model with the provided token
    const hf = new HfInference('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');
    setLLMModel(hf);
    setIsLLMActive(true);

    return () => {
      renderer.dispose();
      if (sceneRef.current) {
        sceneRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  const updateNeuronColors = (activations: number[]) => {
    const maxActivation = Math.max(...activations);
    neuronMeshesRef.current.forEach((mesh, i) => {
      const normalizedActivation = activations[i] / maxActivation;
      const color = new THREE.Color(
        normalizedActivation,
        0,
        1 - normalizedActivation
      );
      const material = mesh.material as THREE.MeshStandardMaterial;
      material.emissive = color;
      material.emissiveIntensity = normalizedActivation;
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || !hnn) return;

    let response: string;
    let activations: number[];

    if (isLLMActive && llmModel) {
      try {
        const llmResponse = await llmModel.textGeneration({
          model: 'facebook/opt-350m',
          inputs: inputText,
          parameters: {
            max_new_tokens: 50,
            temperature: 0.7,
            top_p: 0.95,
            repetition_penalty: 1.1,
          },
        });
        response = llmResponse.generated_text;
      } catch (error) {
        console.error('Error generating LLM response:', error);
        response = "Sorry, I couldn't generate a response.";
      }
      activations = hnn.learn(inputText, response);
    } else {
      const result = hnn.generateResponse(inputText);
      response = result.response;
      activations = result.activations;
    }

    setChatHistory((prev) => [
      ...prev,
      { type: 'user', text: inputText },
      { type: 'bot', text: response },
    ]);
    setInputText('');

    updateNeuronColors(activations);
  };

  const handleLearn = () => {
    if (learnInput.trim() && learnResponse.trim() && hnn) {
      setIsLearning(true);
      const activations = hnn.learn(learnInput, learnResponse);
      updateNeuronColors(activations);
      setLearnInput('');
      setLearnResponse('');
      setIsLearning(false);
      alert('Learning completed');
    }
  };

  const handleSave = () => {
    if (hnn) {
      const knowledge = hnn.exportKnowledge();
      const blob = new Blob([knowledge], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'holographic_nn_knowledge.json';
      a.click();
    }
  };

  const handleLoad = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && hnn) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const knowledge = e.target?.result;
        if (typeof knowledge === 'string') {
          const success = hnn.importKnowledge(knowledge);
          if (success) {
            alert('Knowledge loaded successfully');
          } else {
            alert('Error loading knowledge');
          }
        }
      };
      reader.readAsText(file);
    }
  };

  const handleTrain = () => {
    setIsTraining(true);
    setTrainingProgress(0);

    const trainStep = (i: number) => {
      if (i >= smallDataset.length) {
        setIsTraining(false);
        setTrainingProgress(0);
        alert('Training completed successfully');
        return;
      }

      const item = smallDataset[i];
      if (hnn) {
        const activations = hnn.learn(item.question, item.answer);
        updateNeuronColors(activations);
      }
      setTrainingProgress(Math.round(((i + 1) / smallDataset.length) * 100));

      setTimeout(() => trainStep(i + 1), 100);
    };

    trainStep(0);
  };

  const handleLLM = () => {
    setIsLLMActive(!isLLMActive);
  };

  return (
    <div className="flex h-screen">
      <div className="w-1/2 p-4 overflow-auto">
        <Card>
          <CardHeader>
            <CardTitle>
              Holographic Chat {isLLMActive ? 'with LLM' : ''}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {chatHistory.map((message, index) => (
                <div
                  key={index}
                  className={`p-2 rounded-lg ${
                    message.type === 'user'
                      ? 'bg-blue-100 text-blue-800'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  {message.text}
                </div>
              ))}
            </div>
            <form onSubmit={handleSubmit} className="mt-4 flex space-x-2">
              <Input
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Type your message..."
                className="flex-grow"
              />
              <Button type="submit">Send</Button>
            </form>
          </CardContent>
        </Card>

        <Card className="mt-4">
          <CardHeader>
            <CardTitle>Learn</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <Input
                value={learnInput}
                onChange={(e) => setLearnInput(e.target.value)}
                placeholder="Input to learn"
              />
              <Input
                value={learnResponse}
                onChange={(e) => setLearnResponse(e.target.value)}
                placeholder="Associated response"
              />
              <Button onClick={handleLearn} disabled={isLearning}>
                {isLearning ? 'Learning...' : 'Learn'}
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="mt-4 space-x-2">
          <Button onClick={handleSave}>Save</Button>
          <Button
            onClick={() => document.getElementById('load-knowledge')?.click()}
          >
            Load
          </Button>
          <input
            type="file"
            id="load-knowledge"
            style={{ display: 'none' }}
            onChange={handleLoad}
            accept=".json"
          />
          <Button onClick={handleTrain} disabled={isTraining}>
            {isTraining ? 'Training...' : 'Train'}
          </Button>
          <Button onClick={handleLLM}>
            {isLLMActive ? 'Disable LLM' : 'Enable LLM'}
          </Button>
        </div>

        {isTraining && (
          <div className="mt-4">
            <Progress value={trainingProgress} className="w-full" />
            <p className="text-center mt-2">{trainingProgress}% completed</p>
          </div>
        )}
      </div>

      <div className="w-1/2" ref={sceneRef}></div>
    </div>
  );
}
