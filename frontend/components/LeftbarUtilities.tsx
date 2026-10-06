import { useState } from "react";

import {
  AiOutlineUpload,
  AiOutlineBars,
  AiFillGithub,
  AiOutlineSetting,
  AiOutlineUser,
  AiOutlineLogout,
} from "react-icons/ai";

import Popover from "@mui/material/Popover";
import Typography from "@mui/material/Typography";
import Modal from "@mui/material/Modal";

import Link from "next/link";
import React from "react";
import { FileUploader } from "./LeftbarUploader";

type ModalState = {
  isOpen: boolean;
  content: string | null;
};

export const UtilityBar = ({ lefBarWidth }: { lefBarWidth: number }) => {
  const [anchorEl, setAnchorEl] = useState<HTMLButtonElement | null>(null);
  const [activeButton, setActiveButton] = useState<string | null>(null);

  const handleClick = (
    event: React.MouseEvent<HTMLButtonElement>,
    buttonName: string
  ) => {
    setAnchorEl(event.currentTarget);
    setActiveButton(activeButton === buttonName ? null : buttonName);
  };

  const handleClose = () => {
    setAnchorEl(null);
    setActiveButton(null);
  };

  const open = Boolean(anchorEl);

  const [modalState, setModalState] = useState<ModalState>({
    isOpen: false,
    content: null,
  });

  // Handle opening the modal with specific content
  const handleModalOpen = (content: string) => {
    setModalState({ isOpen: true, content });
  };

  // Handle closing the modal
  const handleModalClose = () => {
    setModalState({ isOpen: false, content: null });
  };

  return (
    <>
      <div className="my-4 mx-2 inline-flex items-center justify-between">
        <div className="inline-flex items-center">
          <button
            className={`flex flex-col items-center px-[7px] py-[3px] text-gray-800 rounded-[9px] ${
              activeButton === "options" ? "bg-gray-200" : "hover:bg-gray-100"
            }`}
            onClick={(e) => handleClick(e, "options")}
          >
            <AiOutlineBars className="cursor-pointer text-[25px]"></AiOutlineBars>
            <p className="cursor-pointer text-[11px]">Options</p>
          </button>
          <Popover
            open={open && activeButton === "options"}
            anchorEl={anchorEl}
            onClose={handleClose}
            anchorOrigin={{
              vertical: "top",
              horizontal: "left",
            }}
            transformOrigin={{
              vertical: "bottom",
              horizontal: "left",
            }}
            sx={{
              transform: "translate(-8px, -6px)",
              "& .MuiPaper-root": {
                width: `${Math.min(lefBarWidth - 16, 180)}px`,
                // height: `88px`,
                boxShadow: "0 0 10px rgba(0, 0, 0, 0.2)", // Custom shadow
                borderRadius: "9px", // Rounded corners
                backgroundColor: "rgba(0, 0, 0, 0.6)", // Semi-transparent white background
                backdropFilter: "blur(9px)",
              },
            }}
          >
            <Typography component={"span"} sx={{ py: "20px" }}>
              <div className="flex flex-col p-2 text-sm text-white">
                <button
                  className="w-ful px-1 py-[8px] rounded-[9px] text-sm hover:bg-gray-900"
                  onClick={() => handleModalOpen("My account")}
                >
                  <div className="group relative inline-flex items-center w-full">
                    <AiOutlineUser className="absolute left-1 text-xl"></AiOutlineUser>
                    <p className="pl-9 whitespace-nowrap text-sm text-ellipsis overflow-hidden cursor-pointer">
                      My account
                    </p>
                  </div>
                </button>

                <button
                  className="w-ful px-1 py-[8px] rounded-[9px] text-sm hover:bg-gray-900"
                  onClick={() => handleModalOpen("Settings")}
                >
                  <div className="group relative inline-flex items-center w-full">
                    <AiOutlineSetting className="absolute left-1 text-xl"></AiOutlineSetting>
                    <p className="pl-9 whitespace-nowrap text-sm text-ellipsis overflow-hidden cursor-pointer">
                      Settings
                    </p>
                  </div>
                </button>

                <Modal open={modalState.isOpen} onClose={handleModalClose}>
                  <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 bg-white shadow-custom_focus p-4 rounded-[9px]">
                    {modalState.content === "My account" && (
                      <div>
                        <h2 className="text-lg font-semibold">My Account</h2>
                        <div className="mt-2">
                          Duis mollis, est non commodo luctus, nisi erat
                          porttitor ligula.
                        </div>
                      </div>
                    )}
                    {modalState.content === "Settings" && (
                      <div>
                        <h2 className="text-lg font-semibold">Settings</h2>
                        <div className="mt-2">
                          Duis mollis, est non commodo luctus, nisi erat
                          porttitor ligula.
                        </div>
                      </div>
                    )}
                  </div>
                </Modal>

                <button className="w-ful px-1 py-[8px] rounded-[9px] text-sm hover:bg-gray-900">
                  <div className="group relative inline-flex items-center w-full">
                    <AiOutlineLogout className="absolute left-1 text-xl"></AiOutlineLogout>
                    <p className="pl-9 whitespace-nowrap text-sm text-ellipsis overflow-hidden cursor-pointer">
                      Logout
                    </p>
                  </div>
                </button>
              </div>
            </Typography>
          </Popover>

          <Link
            href="https://github.com/junruxiong/IncarnaMind"
            target="_blank"
            rel="noopener noreferrer"
            className={`flex flex-col items-center px-[7px] py-[3px] text-gray-800 rounded-[9px] ${
              activeButton === "github" ? "bg-gray-200" : "hover:bg-gray-100"
            }`}
            // onClick={(e) => handleClick(e, "github")}
          >
            <AiFillGithub className="cursor-pointer text-[25px]"></AiFillGithub>
            <p className="cursor-pointer text-[11px]">GitHub</p>
          </Link>
        </div>

        <button
          className={`flex flex-col items-center px-[7px] py-[3px] text-gray-800 rounded-[9px] ${
            activeButton === "upload" ? "bg-gray-200" : "hover:bg-gray-100"
          }`}
          onClick={(e) => handleClick(e, "upload")}
        >
          <AiOutlineUpload className="cursor-pointer text-[25px]"></AiOutlineUpload>
          <p className="cursor-pointer text-[11px]">Upload</p>
        </button>
        <Popover
          open={open && activeButton === "upload"}
          anchorEl={anchorEl}
          onClose={handleClose}
          anchorOrigin={{
            vertical: "top",
            horizontal: "right",
          }}
          transformOrigin={{
            vertical: "bottom",
            horizontal: "right",
          }}
          sx={{
            transform: "translate(-8px, -6px)",
            "& .MuiPaper-root": {
              width: `${lefBarWidth - 16}px`,
              // height: `88px`,
              boxShadow: "0 0 10px rgba(0, 0, 0, 0.1)", // Custom shadow
              borderRadius: "9px", // Rounded corners
              backgroundColor: "rgba(255, 255, 255, 0.6)", // Semi-transparent white background
              backdropFilter: "blur(9px)",
            },
          }}
        >
          <Typography component={"span"} sx={{ py: "20px" }}>
            <FileUploader></FileUploader>
          </Typography>
        </Popover>
      </div>
    </>
  );
};
