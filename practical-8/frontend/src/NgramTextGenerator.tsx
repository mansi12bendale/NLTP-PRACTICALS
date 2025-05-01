import { useState } from "react";
import axios from "axios";

export default function NgramTextGenerator() {
  const [text, setText] = useState("");
  const [n, setN] = useState(2);
  const [numWords, setNumWords] = useState(20);
  const [seed, setSeed] = useState("");
  const [generatedText, setGeneratedText] = useState("");
  const [loading, setLoading] = useState(false);

  const handleTrain = async () => {
    try {
      setLoading(true);
      await axios.post("http://localhost:5000/train", {
        text,
        n,
      });
      alert(`${n}-gram model trained successfully.`);
    } catch (error) {
      alert("Training failed. Check console for details.");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    try {
      setLoading(true);
      const response = await axios.post("http://localhost:5000/generate", {
        num_words: numWords,
        seed,
      });
      setGeneratedText(response.data.generated_text);
    } catch (error) {
      alert("Text generation failed.");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <h1 className="text-3xl font-bold text-center">N-gram Text Generator</h1>

      <div className="space-y-2">
        <label htmlFor="text" className="block font-medium text-gray-700">
          Training Text
        </label>
        <textarea
          id="text"
          rows={6}
          className="w-full p-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste or write your training corpus here..."
        />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div>
          <label htmlFor="n" className="block font-medium text-gray-700">
            N-gram (n)
          </label>
          <input
            id="n"
            type="number"
            min={1}
            max={5}
            className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={n}
            onChange={(e) => setN(parseInt(e.target.value))}
          />
        </div>

        <div>
          <label htmlFor="numWords" className="block font-medium text-gray-700">
            Generate Words
          </label>
          <input
            id="numWords"
            type="number"
            min={1}
            max={100}
            className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={numWords}
            onChange={(e) => setNumWords(parseInt(e.target.value))}
          />
        </div>

        <div>
          <label htmlFor="seed" className="block font-medium text-gray-700">
            Seed Phrase
          </label>
          <input
            id="seed"
            type="text"
            className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            placeholder="Optional seed words"
          />
        </div>
      </div>

      <div className="flex gap-4">
        <button
          onClick={handleTrain}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-4 py-2 rounded-md disabled:opacity-50"
        >
          {loading ? "Training..." : "Train Model"}
        </button>
        <button
          onClick={handleGenerate}
          disabled={loading}
          className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold px-4 py-2 rounded-md disabled:opacity-50"
        >
          {loading ? "Generating..." : "Generate Text"}
        </button>
      </div>

      {generatedText && (
        <div className="mt-6 p-4 border border-gray-300 rounded-md bg-gray-50">
          <h2 className="text-xl font-semibold mb-2">Generated Text:</h2>
          <p className="whitespace-pre-wrap text-gray-700">{generatedText}</p>
        </div>
      )}
    </div>
  );
}
