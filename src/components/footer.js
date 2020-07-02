import React from "react"

import style from "./footer.module.css"

const Footer = ({ siteTitle }) => (
  <footer className={style.colophon}>
    {new Date().getFullYear()} {siteTitle}
  </footer>
)

export default Footer
