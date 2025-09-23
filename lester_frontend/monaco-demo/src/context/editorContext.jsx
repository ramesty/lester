import React, { useRef, createContext, useContext } from "react";
import { setDecorations, setHighlightMap, setInputCode } from "../store/editorSlice";
import { useDispatch, useSelector } from "react-redux";

const EditorContext = createContext(null);

export const EditorProvider = ({ children }) => {
    
    const dispatch = useDispatch();
    const editorRef = useRef(null);
    const monacoRef = useRef(null);

    const colour = useSelector((state) => state.editor.colour);
    const highlightMap = useSelector((state) => state.editor.highlightMap);
    const decorations = useSelector((state) => state.editor.decorations);

    const handleEditorDidMount = (editor, monaco) => {
        editorRef.current = editor;
        monacoRef.current = monaco;
        highlightEditor(highlightMap, editor, monaco)
    };

    const highlightEditor = (newHighlightMap, editor, monaco) => {

        const newDecorations = Object.entries(newHighlightMap).map(([line, clr]) => {
            const lineNum = Number(line);
            const lineLength = editor.getModel().getLineLength(lineNum);
            return {
                range: new monaco.Range(lineNum, 1, lineNum, lineLength + 1),
                options: {
                className: `highlight-${clr}`,
                },
            };
        });
        
        const decorationIds = editor.deltaDecorations([], newDecorations);
        dispatch(setDecorations(decorationIds));
        dispatch(setHighlightMap(newHighlightMap));
    }

    const handleHighlight = () => {

        const editor = editorRef.current;
        const monaco = monacoRef.current;

        const selection = editor.getSelection();
        if (!selection) return;

        const newHighlightMap = { ...highlightMap };
        editor.deltaDecorations(decorations, []);
        dispatch(setDecorations([]));

        for (let line = selection.startLineNumber; line <= selection.endLineNumber; line++) {newHighlightMap[line] = colour;}

        highlightEditor(newHighlightMap, editor, monaco)
    };

    const clearHighlights = () => {

        const editor = editorRef.current;
        if (editor) {
            editor.deltaDecorations(decorations, []);
            dispatch(setDecorations([]));
            dispatch(setHighlightMap({}));
        }
    };

    const handleCodeChange = (updatedCode) => {
        dispatch(setInputCode(updatedCode));
    };

    const handleInputChange = (index, value) => {
        const updatedInputs = [...inputs];
        updatedInputs[index] = value;
        dispatch(setInputs(updatedInputs));
    };

    return (
        <EditorContext.Provider value={{ editorRef, monacoRef, handleEditorDidMount, handleHighlight, clearHighlights, handleCodeChange, handleInputChange }}>
        {children}
        </EditorContext.Provider>
    );
};

export const useEditor = () => useContext(EditorContext);
