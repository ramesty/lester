import { createSlice } from "@reduxjs/toolkit";
import { 
  messy_original_pipeline,
  dataprep_input_arg_names,
  dataprep_input_schemas,
  dataprep_output_columns,
  featurisation_input_schema,
  initialHighlightMap,
} from "../assets/input_data";

  // Input Variables
  // const [inputs, setInputs] = useState([dataprep_input_arg_names, dataprep_input_schemas, dataprep_output_columns, featurisation_input_schema]);
  // const [inputCode, setInputCode] = useState(messy_original_pipeline);
  // const [decorations, setDecorations] = useState(initialEditorDecorations);
  // const [highlightMap, setHighlightMap] = useState(initialHighlightMap);
  // const [testSelection, setTestSelection] = useState("no_selection")
  // const [regenerateSelection, setRegenerateSelection] = useState("no_selection") 
  // const [response, setResponse] = useState([]);
  // const [responseDecorations, setResponseDecorations] = useState([]);
  // const [responseCode, setResponseCode] = useState("");
  // const [lineColourMap, setLineColourMap] = useState([]);
  // const [testResponse, setTestResponse] = useState("")
  // const [regenerateResponse, setRegenerateResponse] = useState([])
  // const [loading, setLoading] = useState(false);

const editorSlice = createSlice({
  name: "editor",
  initialState: {
    inputs: [dataprep_input_arg_names, dataprep_input_schemas, dataprep_output_columns, featurisation_input_schema],
    inputCode: messy_original_pipeline,
    decorations: [],
    highlightMap: initialHighlightMap,
    colour: "no_selection",
    testSelection: "no_selection",
    regenerateSelection: "no_selection",
    validCode: true,
    submitError: '',
    submitErrorMsg: [],
    valid: Array(4).fill(true),
    response: [],
    responseDecorations: [],
    responseCode: "\n\n\n",
    responseHighlightMap: [],
    testResponse: "",
    regenerateResponse: [],
    loading: false,
  },

  reducers: {
    setInputs: (state, action) => { state.inputs = action.payload; },
    setInputCode: (state, action) => { state.inputCode = action.payload; },
    setDecorations: (state, action) => { state.decorations = action.payload; },
    setHighlightMap: (state, action) => { state.highlightMap = action.payload; },
    setColour: (state, action) => { state.colour = action.payload; },

    // For testing and regeneration in the sidebar
    setTestSelection: (state, action) => { state.testSelection = action.payload; },
    setRegenerateSelection: (state, action) => { state.regenerateSelection = action.payload; },

    // Error checking on code submission
    setValidCode: (state, action) => { state.validCode = action.payload; },
    setSubmitError: (state, action) => { state.submitError = action.payload; },
    setSubmitErrorMsg: (state, action) => { state.submitErrorMsg = action.payload; },
    setValid: (state, action) => { state.valid = action.payload; },

    // Response states
    setResponse: (state, action) => { state.response = action.payload; },
    setResponseDecorations: (state, action) => { state.responseDecorations = action.payload; },
    setResponseCode: (state, action) => { state.responseCode = action.payload; },
    setResponseHighlightMap:  (state, action) => { state.responseHighlightMap = action.payload; },

    // Variables for testing and regeneration
    setTestResponse: (state, action) => { state.testResponse = action.payload; },
    setRegenerateResponse: (state, action) => { state.regenerateResponse = action.payload; },
    setLoading:  (state, action) => { state.loading = action.payload; },
  },
});

export const { setInputs, setInputCode, setDecorations, setHighlightMap, setColour, setTestSelection, 
setRegenerateSelection, setValidCode, setSubmitError, setSubmitErrorMsg, setValid,
setResponse, setResponseDecorations, setResponseCode, setResponseHighlightMap, setTestResponse, setRegenerateResponse, setLoading,
} = editorSlice.actions;

export default editorSlice.reducer;