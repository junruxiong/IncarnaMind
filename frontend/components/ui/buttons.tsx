import * as React from "react";
import { styled } from "@mui/material/styles";
import Switch, { SwitchProps } from "@mui/material/Switch";

export const IOSSwitch = styled((props: SwitchProps) => (
  <Switch focusVisibleClassName=".Mui-focusVisible" disableRipple {...props} />
))(({ theme }) => ({
  width: 21, // Half of the original 42
  height: 13, // Half of the original 26
  padding: 0,
  "& .MuiSwitch-switchBase": {
    padding: 0,
    margin: 1, // Adjusted for the smaller size
    transitionDuration: "300ms",
    "&.Mui-checked": {
      transform: "translateX(8px)", // Half of the original 16px
      color: "#fff",
      "& + .MuiSwitch-track": {
        backgroundColor: theme.palette.mode === "dark" ? "#2abb40" : "#65C466",
        opacity: 1,
        border: 0,
      },
      "&.Mui-disabled + .MuiSwitch-track": {
        opacity: 0.5,
      },
    },
    "&.Mui-focusVisible .MuiSwitch-thumb": {
      color: "#33cf4d",
      border: "3px solid #fff", // Adjusted for the smaller size
    },
    "&.Mui-disabled .MuiSwitch-thumb": {
      color:
        theme.palette.mode === "light"
          ? theme.palette.grey[100]
          : theme.palette.grey[600],
    },
    "&.Mui-disabled + .MuiSwitch-track": {
      opacity: theme.palette.mode === "light" ? 0.7 : 0.3,
    },
  },
  "& .MuiSwitch-thumb": {
    boxSizing: "border-box",
    width: 11, // Half of the original 22
    height: 11, // Half of the original 22
  },
  "& .MuiSwitch-track": {
    borderRadius: 13 / 2, // Half of the original 26 / 2
    backgroundColor: theme.palette.mode === "light" ? "#d8d8da" : "#39393D",
    opacity: 1,
    transition: theme.transitions.create(["background-color"], {
      duration: 500,
    }),
  },
}));
