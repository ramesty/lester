import React, { useRef, useState, useEffect } from "react";
import Sidebar from "../components/Sidebar";
import Loading from "../components/Loading";
import CodeEditor from "../components/CodeEditor";
import ResponseCodeEditor from "../components/responseCodeEditor";
import DisplayBox from "../components/DisplayBox";
import { useResponseData } from "../hooks/useResponseData";
import EditorsLayout from "../layouts/EditorsLayout";
import MainLayout from "../layouts/MainLayout";
import PageLayout from "../layouts/PageLayout";
import Navbar from "../components/NavBar";


// redux framework

function Home() {

  return (
  <PageLayout>
    <Navbar/>
    <Sidebar/>
    <MainLayout>
      <EditorsLayout>
          <CodeEditor />
          <ResponseCodeEditor/>
      </EditorsLayout>
      <DisplayBox/>
    </MainLayout>
    <Loading />
    
  </PageLayout>
  );

}

export default Home;
