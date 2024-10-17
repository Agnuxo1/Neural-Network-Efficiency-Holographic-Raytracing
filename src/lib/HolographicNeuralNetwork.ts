import * as THREE from 'three'

export class HolographicNeuralNetwork {
  numNeurons: number
  spaceSize: number
  rayIntensity: number
  neurons: any[]
  knowledgeBase: Record<string, any>

  constructor(numNeurons: number, spaceSize: number, rayIntensity: number) {
    this.numNeurons = numNeurons
    this.spaceSize = spaceSize
    this.rayIntensity = rayIntensity
    this.neurons = this.initializeNeurons()
    this.knowledgeBase = this.loadKnowledge()
  }

  initializeNeurons() {
    return Array.from({ length: this.numNeurons }, () => ({
      position: new THREE.Vector3(
        Math.random() * this.spaceSize,
        Math.random() * this.spaceSize,
        Math.random() * this.spaceSize
      ),
      activation: 0
    }))
  }

  simulateLightPropagation(inputText: string) {
    const hashInput = inputText.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % this.numNeurons
    const initialPosition = this.neurons[hashInput].position

    this.neurons.forEach(neuron => {
      const distance = neuron.position.distanceTo(initialPosition)
      neuron.activation = Math.exp(-distance / this.spaceSize) * this.rayIntensity
    })

    return this.neurons.map(n => n.activation)
  }

  learn(inputText: string, response: string) {
    const activations = this.simulateLightPropagation(inputText)
    const topNeurons = activations
      .map((a, i) => ({ index: i, activation: a }))
      .sort((a, b) => b.activation - a.activation)
      .slice(0, 10)
      .map(n => n.index)
    this.knowledgeBase[inputText] = { response, neurons: topNeurons }
    this.saveKnowledge()
    return activations
  }

  generateResponse(inputText: string) {
    const activations = this.simulateLightPropagation(inputText)
    const topNeurons = activations
      .map((a, i) => ({ index: i, activation: a }))
      .sort((a, b) => b.activation - a.activation)
      .slice(0, 5)
      .map(n => n.index)

    const responses = Object.entries(this.knowledgeBase)
      .filter(([, data]) => data.neurons.some(n => topNeurons.includes(n)))
      .map(([, data]) => data.response)

    return {
      response: responses.length > 0
        ? responses[Math.floor(Math.random() * responses.length)]
        : "No suitable response found.",
      activations: activations
    }
  }

  saveKnowledge() {
    localStorage.setItem('knowledgeBase', JSON.stringify(this.knowledgeBase))
  }

  loadKnowledge() {
    const knowledge = localStorage.getItem('knowledgeBase')
    return knowledge ? JSON.parse(knowledge) : {}
  }

  exportKnowledge() {
    return JSON.stringify(this.knowledgeBase)
  }

  importKnowledge(knowledge: string) {
    try {
      this.knowledgeBase = JSON.parse(knowledge)
      this.saveKnowledge()
      return true
    } catch (error) {
      console.error("Error importing knowledge:", error)
      return false
    }
  }
}