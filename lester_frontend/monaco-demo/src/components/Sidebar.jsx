import { useSelector } from "react-redux";
import InputPanel from "./InputPanel";
import TestPanel from "./TestPanel";
import SidebarLayout from "../layouts/SidebarLayout";
import { Disclosure, DisclosureButton, DisclosurePanel } from '@headlessui/react'
import { ChevronDownIcon } from '@heroicons/react/20/solid'

function Sidebar(){
    
    const responseCode = useSelector((state) => state.editor.inputCode)
    

    return(
        <SidebarLayout>

            <InputPanel></InputPanel>
            <TestPanel></TestPanel>

            {/* <Disclosure as="div" className="p-3 bg-neutral-950 rounded m-1" defaultOpen={true}>
                <DisclosureButton className="group flex w-full items-center justify-between">
                    <span className="text-lg/6 font-medium text-white group-data-hover:text-white/80">
                    Input Panel
                    </span>
                    <ChevronDownIcon className="size-5 fill-white/60 group-data-hover:fill-white/50 group-data-open:rotate-180" />
                </DisclosureButton>
                <DisclosurePanel className="mt-2 text-sm/5 text-white/50">
                    <InputPanel/>
                </DisclosurePanel>
            </Disclosure>

            <Disclosure as="div" className="p-3 bg-neutral-950 rounded m-1">
                <DisclosureButton className="group flex w-full items-center justify-between">
                    <span className="text-lg/6 font-medium text-white group-data-hover:text-white/80">
                    Test Panel
                    </span>
                    <ChevronDownIcon className="size-5 fill-white/60 group-data-hover:fill-white/50 group-data-open:rotate-180" />
                </DisclosureButton>
                <DisclosurePanel className="mt-2 text-sm/5 text-white/50">
                    <TestPanel/>
                </DisclosurePanel>
            </Disclosure> */}
            
        </SidebarLayout>
    )
}

export default Sidebar