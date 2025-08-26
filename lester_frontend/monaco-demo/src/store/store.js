import { configureStore } from "@reduxjs/toolkit";
// import uiReducer from "./uiSlice";
// import authReducer from "./authSlice";
import editorReducer from "./editorSlice";

export const store = configureStore({
  reducer: {
    // ui: uiReducer,
    // auth: authReducer,
    editor: editorReducer,
  },
});