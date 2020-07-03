/**
 * Layout component that queries for data
 * with Gatsby's useStaticQuery component
 *
 * See: https://www.gatsbyjs.org/docs/use-static-query/
 */

import React from "react"
//import { useStaticQuery, graphql } from "gatsby"

import Header from "./header"
import Footer from "./footer"

// Styles
import "../styles/reset.css"
import "../styles/accessibility.css"
import "../fonts/fonts.css"
import style from "./layout.module.css"

const Layout = ({ children }) => {
  return (
    <>
      <a className="skip-link screen-reader-text" href="#primary">
        Skip to the content
      </a>
      <Header siteTitle="JumpML" siteDesc="Let's Build Intelligent Things!" />
      <main id="primary" className={style.site_main}>
        {children}
      </main>

      <Footer siteTitle="JumpML" />
    </>
  )
}

export default Layout
