import React, { useRef, useState, useEffect } from "react";
import { callBackend } from '../utils/api';
import Sidebar from "../components/Sidebar";
import Codebox from "../components/Codebox";
import Loading from "../components/Loading";
import SplitPane from "react-split-pane";

import "../App.css";

function Home() {

  // Variables used in the input editor
  const editorRef = useRef(null);
  const monacoRef = useRef(null);

  // Input Variables
  const [inputs, setInputs] = useState(Array(4).fill());
  const [inputCode, setInputCode] = useState("def foo():\n    return 42\n\nprint(foo())");
  const [colour, setColour] = useState("green");
  const [decorations, setDecorations] = useState([]);
  const [highlightMap, setHighlightMap] = useState({});
  
  // Error checking on code submission
  const [validCode, setValidCode] = useState(true)
  const [submitError, setSubmitError] = useState('');
  const [submitErrorMsg, setSubmitErrorMsg] = useState([]);

  // waiting and receiving response 
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState([]);


  // Variables used in the output editor
  const responseEditorRef = useRef(null);
  const responseMonacoRef = useRef(null);

  // Decorations and colours for synthesized code
  const [responseDecorations, setResponseDecorations] = useState([]);
  const [responseCode, setResponseCode] = useState("def foo():\n    return 42\n\nprint(foo())");
  const [lineColourMap, setLineColourMap] = useState([]);

  // Variables for testing and regeneration
  const [testSelection, setTestSelection] = useState("no_selection")
  const [regenerateSelection, setRegenerateSelection] = useState("no_selection") 

  useEffect(() => {

    const lines = response.map(item => item.line);
    const colours = response.map(item => item.colour);

    setResponseCode(lines.join("\n"));
    setLineColourMap(colours)


  }, [response]);

  useEffect(() => {

    if (responseEditorRef.current && responseMonacoRef.current) {
      decorateResponseEditor(responseEditorRef.current, responseMonacoRef.current, lineColourMap);
    }

  }, [lineColourMap]);

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

  const handleCodeChange = (updatedCode) => {
    console.log("Changing input code...")
    setCode(updatedCode);
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
      newHighlightMap[line] = colour;
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
      
      <Sidebar inputs={inputs}
        setInputs={setInputs}
        colour={colour}
        setColour={setColour}
        testSelection={testSelection}
        setTestSelection={setTestSelection}
        regenerateSelection={regenerateSelection}
        setRegenerateSelection={setRegenerateSelection}
        handleHighlight={handleHighlight}
        clearHighlights={clearHighlights}
        submitCode={submitCode}
        testCodeSubmit={testCodeSubmit}
        regenerateCodeSubmit={regenerateCodeSubmit}
        submitError={submitError}
        submitErrorMsg={submitErrorMsg}
        />



      <div className="col-span-5">

        <div className="grid grid-cols-2 h-1/2 gap-4">
            <Codebox validCode={validCode} title={"Original Code"} code={inputCode} handleCodeChange={handleCodeChange} editorRef={editorRef} monacoRef={monacoRef} isReadOnly={true} />
            <Codebox validCode={validCode} title={"Synthesized Code"} code={responseCode} handleCodeChange={handleCodeChange} editorRef={responseEditorRef} monacoRef={responseMonacoRef} isReadOnly={true}/>

        </div >
          
        <div className="grid grid-cols-1 h-1/2 gap-4">
            <div><h1>hellow</h1></div>
        </div>
          
      </div>


      <Loading loading={loading} />
    </div>
  </div>
  );

}

export default Home;
