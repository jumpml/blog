import React from "react"
import Layout from "../components/layout"
import SEO from "../components/seo"
import style from "./about.module.css"
import { Link } from "gatsby"

const AboutPage = () => {
  return (
    <Layout>
      <SEO
        title="About JumpML"
        description="Information about JumpML"
        image="/jumpML.svg"
        pathname="/about"
        // Boolean indicating whether this is an article:
        // article
      />
      <section className={style.wrapper}>
        <h2 className={style.heading}>About Us</h2>
        <h1> Background </h1>
        <p>
          JumpML is an audio machine learning (ML) algorithms software company in sunny Torrance (CA).
          JumpML was founded in 2020.

          <li className={style.flexcontainer}>
           
          <ul className={style.ul}>
          Brave Founder
          <img
           src="founder.png"
           width="366"
           height="374"
           alt="Ragh"
           className={style.site_logo}
           />
           </ul>
           <ul className={style.ul}>
           Story 
          <li className={style.flexitem}> PhD in Electrical Engineering at UCLA</li>   
          <li className={style.flexitem}> Start-up and big company experience Vxtel/Intel, Beats/Apple</li>
          <li className={style.flexitem}> Published 20+ research papers and patents</li>      
          <li className={style.flexitem}> Experience shipping high-impact algorithm software at scale</li>
          <li className={style.flexitem}> Try my adaptive ANC algorithm on Airpods Pro!</li>    
          </ul>
          </li>
         
          We want to help people build useful things. Our algorithms help solve challenging 
          and impactful problems in the speech and audio domain. 
        </p>
        <h1>More Information</h1>
        <p>
        Please checkout our <Link to="/products">
        Products
              </Link>{" "} page for more information on our current offerings. 
        </p>
        <p>
        For technology demos and specific use cases, please checkout our <Link to="/solutions">
        Use Cases</Link>{" "} page.
        </p>

        <p>
          For questions and more details, please contact us at{" "}
          <a href="mailto:info@jumpml.com">JumpML email.</a>
        </p>
      </section>
    </Layout>
  )
}

export default AboutPage
