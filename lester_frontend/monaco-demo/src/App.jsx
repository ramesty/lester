import React, { useRef, useState, useEffect } from "react";
import { callBackend } from './utils/api';
import Editor from "@monaco-editor/react";
import "./App.css";

function App() {

  // Variables used in the input editor
  const editorRef = useRef(null);
  const monacoRef = useRef(null);

  const [code, setCode] = useState("def foo():\n    return 42\n\nprint(foo())");
  const [color, setColor] = useState("green");

  const labels = [
    "Dataprep input arg names",
    "Dataprep input schemas",
    "Dataprep output columns",
    "Featurisation input schema",
  ];

  const [inputs, setInputs] = useState(Array(labels.length).fill());
  const [valid, setValid] = useState(Array(labels.length).fill(true));
  const [validCode, setValidCode] = useState(true)

  const [submitError, setSubmitError] = useState('');
  const [submitErrorMsg, setSubmitErrorMsg] = useState([]);

  const [decorations, setDecorations] = useState([]);
  const [highlightMap, setHighlightMap] = useState({});
  const [response, setResponse] = useState([]);
  const [loading, setLoading] = useState(false);

  // Variables used in the output editor
  const responseEditorRef = useRef(null);
  const responseMonacoRef = useRef(null);

  const [responseDecorations, setResponseDecorations] = useState([]);
  const [fullCode, setFullCode] = useState("def foo():\n    return 42\n\nprint(foo())");
  const [lineColourMap, setLineColourMap] = useState([]);

  // Variable used for testing
  const [testSelection, setTestSelection] = useState("no_selection")
  const testLabels = [
    "no_selection",
    "test_data_preperation",
    "test_featurisation",
    "test_model"
  ]

  // Variables used for regeneration
  const [regenerateSelection, setRegenerateSelection] = useState("no_selection") 
  const regenerateLabels = [
    "no_selection",
    "regenerate_data_preperation",
    "regenerate_featurisation",
    "regenerate_model"
  ]

  // When a response is received, we update the full code (code shown on response window)
  // as well as the colour map. Changing the colour map calls a seperate useffect, which decorates
  // the respons window.

  useEffect(() => {

    const lines = response.map(item => item.line);
    const colours = response.map(item => item.colour);

    setFullCode(lines.join("\n"));
    setLineColourMap(colours)


  }, [response]);

  useEffect(() => {

    if (responseEditorRef.current && responseMonacoRef.current) {
      decorateResponseEditor(responseEditorRef.current, responseMonacoRef.current, lineColourMap);
    }

  }, [lineColourMap]);

  const handleEditorDidMount = (editor, monaco) => {

    console.log("editor mount")

    editorRef.current = editor;
    monacoRef.current = monaco;
  };

  const handleResponseEditorMount = (editor, monaco) => {

    console.log("response editor mount")

    responseEditorRef.current = editor;
    responseMonacoRef.current = monaco;

  };

  const decorateResponseEditor = (editor, monaco, lineColours) => {

    editor.deltaDecorations(responseDecorations, []);
    setResponseDecorations([]);

    const newDecorations = lineColours.map((colour, index) => ({
      range: new monaco.Range(index + 1, 1, index + 1, 1),
      options: {
        isWholeLine: true,
        className: `highlight-${colour}`,
      },
    }));

    const decorationIds = editor.deltaDecorations([], newDecorations);
    setResponseDecorations(decorationIds);


  };

  const handleHighlight = () => {

    console.log("handle highlight")

    const editor = editorRef.current;
    const monaco = monacoRef.current;

    const selection = editor.getSelection();
    if (!selection) return;

    const newHighlightMap = { ...highlightMap };
    editor.deltaDecorations(decorations, []);
    setDecorations([]);

    for (
      let line = selection.startLineNumber;
      line <= selection.endLineNumber;
      line++
    ) {
      newHighlightMap[line] = color;
    }

    const newDecorations = Object.entries(newHighlightMap).map(([line, clr]) => ({
      range: new monaco.Range(Number(line), 1, Number(line), 1),
      options: {
        isWholeLine: true,
        className: `highlight-${clr}`,
      },
    }));

    const decorationIds = editor.deltaDecorations([], newDecorations);
    setDecorations(decorationIds);
    setHighlightMap(newHighlightMap);
  };

  const clearHighlights = () => {

    console.log("clear highlights")

    const editor = editorRef.current;
    if (editor) {
      editor.deltaDecorations(decorations, []);
      setDecorations([]);
      setHighlightMap({});
    }
  };

  const validateArray = (str) => {
    // try {
    //   const parsed = JSON.parse(str);
    //   const is1D = Array.isArray(parsed) && !Array.isArray(parsed[0]);
    //   const is2D = Array.isArray(parsed) && parsed.every((el) => Array.isArray(el));
    //   return is1D || is2D;
    // } catch {
    //   return false;
    // }
    return true
  };

  const handleInputChange = (index, value) => {

    console.log("Changing inputs...")

    const updatedInputs = [...inputs];
    updatedInputs[index] = value;
    setInputs(updatedInputs);

  };

  const handleCodeChange = (updatedCode) => {
    console.log("Changing input code...")
    setCode(updatedCode);
  };

  const regenerateCodeSubmit = async () => {

    if (regenerateSelection==="no_selection") {alert("No regeneration option selected!"); return}
    setLoading(true)

    try{
      const result = await callBackend({
        url : "http://127.0.0.1:8000/regenerate_stage/" + regenerateSelection
      })
      setResponse(result)
    } catch (error) {
      console.log("Backend call failed: ", error)
    } finally {
      setLoading(false)
    }
  }

  const testCodeSubmit = async () => {

    if (testSelection === "no_selection") {alert("No test selected!"); return}
    setLoading(true);

    try {
      const result = await callBackend({
        url: "http://127.0.0.1:8000/test_stage/" + testSelection
      });
      console.log(result)

    } catch (error) {
      console.error("Backend call failed:", error);
    } finally {
      setLoading(false);
    }
  }

  const submitCode = async () => {
    
    if (loading) return;
    console.log('Trying to Submit...');

    const validationResults = inputs.map(validateArray);
    setValid(validationResults);

    const allValid = validationResults.every(Boolean);
    const highlightValues = new Set(Object.values(highlightMap));

    // Check highlighting coverage
    if (highlightValues.size < 3) {
      setSubmitError(true);
      setSubmitErrorMsg('Complete Highlighting was not Applied');
      setValidCode(false)
      return;
    }

    // Check input validity
    if (!allValid) {
      const invalidIndexes = validationResults
        .map((valid, idx) => (valid ? null : idx))
        .filter(idx => idx !== null);

      setSubmitError(true);
      setSubmitErrorMsg(`Invalid input at index(es): ${invalidIndexes.join(', ')}`);
      return;
    }

    setLoading(true);
    setSubmitError(false);
    setSubmitErrorMsg('');

    try {
      const result = await callBackend({
        url: 'http://127.0.0.1:8000/run',
        payload: { code, highlightMap, manualInputs: inputs },
      });
      setResponse(result);
      console.log("Submitted inputs:", inputs.map(v => JSON.parse(v)));

    } catch (error) {
      console.error("Backend call failed:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
  <div className="h-screen w-screen p-4">
    <div className="grid grid-cols-6 gap-4 h-full">
      
      {/* Sidebar */}
      <div className="col-span-1 flex flex-col gap-4 overflow-y-auto p-2 border rounded border-gray-300">
        <h2 className="text-xl font-semibold">Lester</h2>

        <div className="grid grid-cols-1 gap-2">
          {inputs.map((value, index) => (
            <div key={index} className="flex flex-col">
              <label className="mb-1 font-medium text-sm text-gray-700">
                {labels[index]}
              </label>
              <input
                type="text"
                placeholder={value}
                onChange={(e) => handleInputChange(index, e.target.value)}
                className={`p-2 border rounded ${valid[index] ? "border-gray-300" : "border-red-500"}`}
              />
            </div>
          ))}
        </div>

        <div className="flex flex-col gap-2">
          <label>Choose Highlight Color:</label>
          <select value={color} onChange={(e) => setColor(e.target.value)} className="p-2 border rounded">
            <option value="green">Data Preparation - Green</option>
            <option value="yellow">Featurisation - Yellow</option>
            <option value="red">Model Training - Red</option>
          </select>
        </div>

        <div className="flex flex-col gap-2">
          <label>Test Synthesized Stage:</label>
          <select value={testSelection} onChange={(e) => setTestSelection(e.target.value)} className="p-2 border rounded">
            {testLabels.map((value, index) => (
              <option value={value} key={index}>{value}</option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-2">
          <label>Regenerate Synthesized Stage:</label>
          <select value={regenerateSelection} onChange={(e) => setRegenerateSelection(e.target.value)} className="p-2 border rounded">
            {regenerateLabels.map((value, index) => (
              <option value={value} key={index}>{value}</option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-2">
          <button onClick={handleHighlight} className="p-2 bg-blue-500 text-white rounded">Highlight Selection</button>
          <button onClick={clearHighlights} className="p-2 bg-gray-500 text-white rounded">Clear Highlights</button>
          <button onClick={submitCode} className="p-2 bg-green-600 text-white rounded">Submit Code</button>
          <button onClick={testCodeSubmit} className="p-2 bg-green-600 text-white rounded">Test Code</button>
          <button onClick={regenerateCodeSubmit} className="p-2 bg-green-600 text-white rounded">Regenerate Code</button>
        </div>

        {submitError && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-2 rounded shadow-md z-50">
            <strong className="font-semibold">Input Error:</strong> {submitErrorMsg}
          </div>
        )}
      </div>

      <div className="col-span-5 h-full grid grid-cols-2 gap-4">
        <div className={`flex flex-col h-full p-2 border rounded ${validCode ? "border-gray-300" : "border-red-500"}`}>
          <h2>Input Code</h2>
          <div className="flex-grow">
            <Editor
              defaultLanguage="python"
              defaultValue={code}
              onChange= {handleCodeChange}
              onMount={handleEditorDidMount}
              theme="vs-dark"
              options={{ minimap: { enabled: false } }}
              height="100%"
            />
          </div>
        </div>

        <div className="flex flex-col h-full p-2 border rounded border-gray-300">

          <h2>Synthesized Response</h2>
          
          <div className="flex-grow">
            <Editor
              defaultLanguage="python"
              value={fullCode}
              onMount={handleResponseEditorMount}
              theme="vs-dark"
              options={{ readOnly: true, minimap: { enabled: false } }}
              height="100%"
            />
          </div>
        </div>
      </div>

      {/* Loading Overlay */}
      {loading && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-30">
          <div className="flex flex-col items-center">
            <div className="w-10 h-10 border-4 border-white border-t-transparent rounded-full animate-spin"></div>
            <span className="mt-2 text-white">Loading...</span>
          </div>
        </div>
      )}
    </div>
  </div>
  );

}

export default App;
